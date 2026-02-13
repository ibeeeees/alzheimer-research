"""Multi-phase trainers for the Alzheimer's multi-task system.

Phase1Trainer: MRI encoder pretraining (ordinal CDR only, cross-sectional)
Phase2Trainer: Full multi-task + longitudinal + cross-cohort alignment
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.full_model import AlzheimerMultiTaskModel
from models.losses import (
    MultiTaskLoss,
    amyloid_bce_loss,
    coral_ordinal_loss,
    discrete_survival_loss,
)
from models.alignment import ClassConditionedMMD
from .callbacks import CheckpointManager, EarlyStopping


def _set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_lr_lambda(current_step: int, total_steps: int,
                       warmup_steps: int) -> float:
    """Cosine annealing with linear warmup."""
    if current_step < warmup_steps:
        return current_step / max(warmup_steps, 1)
    progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════════════
# Phase 1: MRI Pretraining (ordinal CDR only)
# ══════════════════════════════════════════════════════════════════════

class Phase1Trainer:
    """Single-task MRI encoder pretraining on ordinal CDR.

    Cross-sectional: each sample is a single volume + label.
    """

    def __init__(
        self,
        model: AlzheimerMultiTaskModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        # Only train MRI encoder + ordinal head in Phase 1
        self.optimizer = torch.optim.AdamW(
            [
                {"params": model.mri_encoder.parameters()},
                {"params": model.ordinal_head.parameters()},
            ],
            lr=cfg.phase1_lr,
            weight_decay=cfg.phase1_weight_decay,
        )

        total_steps = (
            len(train_loader) // cfg.phase1_grad_accum_steps * cfg.phase1_epochs
        )
        warmup_steps = int(total_steps * cfg.phase1_warmup_frac)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _cosine_lr_lambda(step, total_steps, warmup_steps),
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
        self.early_stopping = EarlyStopping(patience=cfg.phase1_patience, mode="max")
        self.ckpt_manager = CheckpointManager(cfg.checkpoint_dir, prefix="phase1")

        self.history: List[Dict[str, float]] = []

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            volume = batch["volume"].to(self.device)   # (B, 1, D, H, W)
            label = batch["label"].to(self.device)     # (B,)

            with torch.amp.autocast("cuda", enabled=self.cfg.use_amp):
                out = self.model.forward_phase1(volume)
                loss = coral_ordinal_loss(
                    out["cum_logits"], label, self.cfg.num_classes
                )
                loss = loss / self.cfg.phase1_grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.cfg.phase1_grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item() * self.cfg.phase1_grad_accum_steps
            n_batches += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            volume = batch["volume"].to(self.device)
            label = batch["label"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.cfg.use_amp):
                out = self.model.forward_phase1(volume)
                loss = coral_ordinal_loss(
                    out["cum_logits"], label, self.cfg.num_classes
                )

            preds = (out["cum_logits"] > 0).sum(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())
            total_loss += loss.item()
            n_batches += 1

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()

        from evaluation.metrics import compute_qwk, compute_mae

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_qwk": compute_qwk(labels, preds),
            "val_mae": compute_mae(labels, preds),
        }

    def fit(self, resume: bool = False) -> Dict[str, Any]:
        """Run Phase 1 training loop.

        Args:
            resume: If True, resume from latest checkpoint.

        Returns:
            dict with training history and best metrics.
        """
        _set_seed(self.cfg.seed)
        start_epoch = 0

        if resume and self.ckpt_manager.exists("latest"):
            ckpt = self.ckpt_manager.load(which="latest")
            self.model.load_state_dict(ckpt["model_state"], strict=False)
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed Phase 1 from epoch {start_epoch}")

        best_qwk = -1.0

        for epoch in range(start_epoch, self.cfg.phase1_epochs):
            t0 = time.time()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            elapsed = time.time() - t0

            metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.history.append(metrics)

            qwk = val_metrics["val_qwk"]
            is_best = qwk > best_qwk
            if is_best:
                best_qwk = qwk

            self.ckpt_manager.save(
                state={
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scaler_state": self.scaler.state_dict(),
                },
                metric=qwk,
                epoch=epoch,
                is_best=is_best,
            )

            print(
                f"[Phase1] Epoch {epoch:3d} | "
                f"loss {train_metrics['train_loss']:.4f} | "
                f"val_qwk {qwk:.4f} {'*' if is_best else ' '} | "
                f"val_mae {val_metrics['val_mae']:.3f} | "
                f"{elapsed:.1f}s"
            )

            if self.early_stopping.step(qwk):
                print(f"Early stopping at epoch {epoch}")
                break

        return {"history": self.history, "best_qwk": best_qwk}


# ══════════════════════════════════════════════════════════════════════
# Phase 2: Multi-Task + Longitudinal + Alignment
# ══════════════════════════════════════════════════════════════════════

class Phase2Trainer:
    """Full multi-task training with longitudinal MRI and cross-cohort alignment.

    Alternates between MRI batches (ordinal + survival + amyloid) and
    speech batches (ordinal only), with an alignment loss computed on
    paired embeddings from both batches.
    """

    def __init__(
        self,
        model: AlzheimerMultiTaskModel,
        mri_train_loader: DataLoader,
        mri_val_loader: DataLoader,
        speech_train_loader: Optional[DataLoader],
        speech_val_loader: Optional[DataLoader],
        cfg,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.mri_train_loader = mri_train_loader
        self.mri_val_loader = mri_val_loader
        self.speech_train_loader = speech_train_loader
        self.speech_val_loader = speech_val_loader
        self.cfg = cfg
        self.device = device

        # Differential learning rates
        self.optimizer = torch.optim.AdamW(
            [
                {"params": model.mri_encoder.parameters(), "lr": cfg.phase2_lr_backbone},
                {"params": model.temporal_gru.parameters(), "lr": cfg.phase2_lr_heads},
                {"params": model.speech_encoder.parameters(), "lr": cfg.phase2_lr_heads},
                {"params": model.ordinal_head.parameters(), "lr": cfg.phase2_lr_heads},
                {"params": model.survival_head.parameters(), "lr": cfg.phase2_lr_heads},
                {"params": model.amyloid_head.parameters(), "lr": cfg.phase2_lr_heads},
            ],
            weight_decay=cfg.phase2_weight_decay,
        )

        self.multi_task_loss = MultiTaskLoss(
            init_log_var_ord=cfg.init_log_var_ord,
            init_log_var_surv=cfg.init_log_var_surv,
            init_log_var_amy=cfg.init_log_var_amy,
            alignment_lambda=cfg.alignment_lambda,
        ).to(device)

        # Add multi-task loss params to optimizer
        self.optimizer.add_param_group({
            "params": self.multi_task_loss.parameters(),
            "lr": cfg.phase2_lr_heads,
        })

        steps_per_epoch = len(mri_train_loader)
        total_steps = steps_per_epoch * cfg.phase2_epochs
        warmup_steps = int(total_steps * cfg.phase2_warmup_frac)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _cosine_lr_lambda(step, total_steps, warmup_steps),
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
        self.early_stopping = EarlyStopping(patience=cfg.phase2_patience, mode="max")
        self.ckpt_manager = CheckpointManager(cfg.checkpoint_dir, prefix="phase2")
        self.mmd = ClassConditionedMMD(num_classes=cfg.num_classes)

        self.history: List[Dict[str, float]] = []

    def _alignment_warmup(self, epoch: int) -> float:
        """Linear warmup of alignment lambda over first N epochs."""
        if epoch >= self.cfg.alignment_warmup_epochs:
            return 1.0
        return epoch / max(self.cfg.alignment_warmup_epochs, 1)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.multi_task_loss.train()
        warmup_factor = self._alignment_warmup(epoch)

        running = {"total": 0.0, "ord": 0.0, "surv": 0.0, "align": 0.0}
        n_steps = 0

        speech_iter = (
            iter(self.speech_train_loader)
            if self.speech_train_loader
            else None
        )

        for step, mri_batch in enumerate(self.mri_train_loader):
            self.optimizer.zero_grad()

            # ── MRI branch ──────────────────────────────────────
            volumes = mri_batch["volumes"].to(self.device)
            time_deltas = mri_batch["time_deltas"].to(self.device)
            lengths = mri_batch["lengths"].to(self.device)
            mri_labels = mri_batch["label"].to(self.device)

            has_survival = "event_indicators" in mri_batch
            has_mci = "is_mci" in mri_batch

            with torch.amp.autocast("cuda", enabled=self.cfg.use_amp):
                mri_out = self.model.forward_mri_multitask(
                    volumes, time_deltas, lengths,
                    run_survival=has_survival,
                    run_amyloid=False,  # enable when amyloid labels available
                )

                # Ordinal loss (all MRI subjects)
                loss_ord = coral_ordinal_loss(
                    mri_out["ord_cum_logits"], mri_labels, self.cfg.num_classes
                )

                # Survival loss (MCI subjects only)
                loss_surv = None
                if has_survival and has_mci:
                    mci_mask = mri_batch["is_mci"].to(self.device).bool()
                    if mci_mask.any():
                        ei = mri_batch["event_indicators"].to(self.device)
                        arm = mri_batch["at_risk_mask"].to(self.device)
                        loss_surv = discrete_survival_loss(
                            mri_out["surv_hazard_logits"][mci_mask],
                            ei[mci_mask],
                            arm[mci_mask],
                        )

                # ── Speech branch + alignment ───────────────────
                loss_align = None
                if speech_iter is not None:
                    try:
                        speech_batch = next(speech_iter)
                    except StopIteration:
                        speech_iter = iter(self.speech_train_loader)
                        speech_batch = next(speech_iter)

                    speech_feats = speech_batch["features"].to(self.device)
                    speech_labels = speech_batch["label"].to(self.device)

                    speech_out = self.model.forward_speech(speech_feats)
                    # Add speech ordinal loss to the ordinal total
                    loss_ord_speech = coral_ordinal_loss(
                        speech_out["cum_logits"], speech_labels, self.cfg.num_classes
                    )
                    loss_ord = 0.5 * (loss_ord + loss_ord_speech)

                    # Alignment: MMD between MRI and speech embeddings
                    loss_align = self.mmd(
                        mri_out["embedding"], mri_labels,
                        speech_out["embedding"], speech_labels,
                    )

                # ── Combine ─────────────────────────────────────
                loss_dict = self.multi_task_loss(
                    loss_ord=loss_ord,
                    loss_surv=loss_surv,
                    loss_align=loss_align,
                    alignment_warmup_factor=warmup_factor,
                )
                total_loss = loss_dict["total"]

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            running["total"] += total_loss.item()
            running["ord"] += loss_dict["ord_raw"].item()
            if loss_surv is not None:
                running["surv"] += loss_dict.get("surv_raw", torch.tensor(0.0)).item()
            if loss_align is not None:
                running["align"] += loss_dict.get("align", torch.tensor(0.0)).item()
            n_steps += 1

        return {k: v / max(n_steps, 1) for k, v in running.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        n = 0

        for batch in self.mri_val_loader:
            volumes = batch["volumes"].to(self.device)
            time_deltas = batch["time_deltas"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.cfg.use_amp):
                out = self.model.forward_mri_multitask(
                    volumes, time_deltas, lengths,
                )
                loss = coral_ordinal_loss(
                    out["ord_cum_logits"], labels, self.cfg.num_classes
                )

            preds = (out["ord_cum_logits"] > 0).sum(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            n += 1

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()

        from evaluation.metrics import compute_qwk, compute_mae

        return {
            "val_loss": total_loss / max(n, 1),
            "val_qwk": compute_qwk(labels, preds),
            "val_mae": compute_mae(labels, preds),
        }

    def fit(self, resume: bool = False) -> Dict[str, Any]:
        """Run Phase 2 training loop."""
        _set_seed(self.cfg.seed)
        start_epoch = 0

        if resume and self.ckpt_manager.exists("latest"):
            ckpt = self.ckpt_manager.load(which="latest")
            self.model.load_state_dict(ckpt["model_state"], strict=False)
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scaler.load_state_dict(ckpt["scaler_state"])
            self.multi_task_loss.load_state_dict(ckpt["mtl_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed Phase 2 from epoch {start_epoch}")

        best_qwk = -1.0

        for epoch in range(start_epoch, self.cfg.phase2_epochs):
            t0 = time.time()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            elapsed = time.time() - t0

            metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.history.append(metrics)

            qwk = val_metrics["val_qwk"]
            is_best = qwk > best_qwk
            if is_best:
                best_qwk = qwk

            self.ckpt_manager.save(
                state={
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scaler_state": self.scaler.state_dict(),
                    "mtl_state": self.multi_task_loss.state_dict(),
                },
                metric=qwk,
                epoch=epoch,
                is_best=is_best,
            )

            # Log learned task weights
            w_ord = (0.5 * torch.exp(-self.multi_task_loss.log_var_ord)).item()
            w_surv = (0.5 * torch.exp(-self.multi_task_loss.log_var_surv)).item()

            print(
                f"[Phase2] Epoch {epoch:3d} | "
                f"total {train_metrics['total']:.4f} | "
                f"ord {train_metrics['ord']:.4f} | "
                f"surv {train_metrics['surv']:.4f} | "
                f"val_qwk {qwk:.4f} {'*' if is_best else ' '} | "
                f"w_ord {w_ord:.3f} w_surv {w_surv:.3f} | "
                f"{elapsed:.1f}s"
            )

            if self.early_stopping.step(qwk):
                print(f"Early stopping at epoch {epoch}")
                break

        return {"history": self.history, "best_qwk": best_qwk}
