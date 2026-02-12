"""Unimodal training script for MRI-only or Audio-only Alzheimer's staging."""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config
from models.ordinal_utils import (
    OrdinalHead,
    coral_ordinal_loss,
    categorical_cross_entropy_loss,
    compute_all_metrics,
)
from models.fusion_model import FusionModel


# ---------- Dataset ----------


class EmbeddingDataset(Dataset):
    """Loads pre-extracted embeddings and labels from disk."""

    def __init__(self, embeddings_path, labels_path, modality="mri"):
        data = np.load(embeddings_path)
        self.embeddings = data["embeddings"].astype(np.float32)

        import pandas as pd
        labels_df = pd.read_csv(labels_path)
        self.labels = labels_df["label"].values.astype(np.int64)

        assert len(self.embeddings) == len(self.labels), (
            f"Embedding count ({len(self.embeddings)}) != label count ({len(self.labels)})"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx]),
            torch.tensor(self.labels[idx]),
        )


class SyntheticEmbeddingDataset(Dataset):
    """Synthetic dataset for testing the training pipeline."""

    def __init__(self, n_samples=1000, embed_dim=256, num_classes=4, seed=42):
        rng = np.random.RandomState(seed)
        self.labels = rng.randint(0, num_classes, size=n_samples).astype(np.int64)
        # Create embeddings correlated with labels for non-trivial learning
        self.embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)
        for i in range(n_samples):
            center = self.labels[i] * 0.5
            self.embeddings[i] = rng.randn(embed_dim).astype(np.float32) * 0.3 + center

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx]),
            torch.tensor(self.labels[idx]),
        )


# ---------- Training ----------


def train_one_epoch(model, ordinal_head, dataloader, optimizer, config, loss_type="coral"):
    model.train()
    ordinal_head.train()
    total_loss = 0.0
    n_batches = 0

    for embeddings, labels in dataloader:
        embeddings = embeddings.to(config.device)
        labels = labels.to(config.device)

        optimizer.zero_grad()
        severity = model(embeddings)  # (B, 1)

        if loss_type == "coral":
            logits = ordinal_head(severity)
            loss = coral_ordinal_loss(logits, labels, config.num_classes)
        else:
            loss = categorical_cross_entropy_loss(severity, ordinal_head, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, ordinal_head, dataloader, config, loss_type="coral"):
    model.eval()
    ordinal_head.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_scores = []
    total_loss = 0.0
    n_batches = 0

    for embeddings, labels in dataloader:
        embeddings = embeddings.to(config.device)
        labels = labels.to(config.device)

        severity = model(embeddings)

        if loss_type == "coral":
            logits = ordinal_head(severity)
            loss = coral_ordinal_loss(logits, labels, config.num_classes)
        else:
            loss = categorical_cross_entropy_loss(severity, ordinal_head, labels)

        total_loss += loss.item()
        n_batches += 1

        preds = ordinal_head.predict(severity)
        probs = ordinal_head.class_probabilities(severity)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_scores.append(severity.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_scores = np.concatenate(all_scores)

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_all_metrics(all_labels, all_preds, all_probs, config.num_classes)
    metrics["loss"] = avg_loss
    metrics["severity_scores"] = all_scores

    return metrics


def train_model(
    train_dataset,
    val_dataset,
    config,
    embed_dim=256,
    loss_type="coral",
    fold=None,
    verbose=True,
):
    """Full training loop with early stopping.

    Returns:
        model, ordinal_head, best_metrics, history
    """
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Unimodal: single embedding → severity score
    model = FusionModel(
        embed_dim=embed_dim,
        hidden_dims=config.fusion_hidden_dims,
        dropout=config.fusion_dropout,
        num_modalities=1,
    ).to(config.device)

    ordinal_head = OrdinalHead(num_classes=config.num_classes).to(config.device)

    all_params = list(model.parameters()) + list(ordinal_head.parameters())
    optimizer = Adam(all_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    if config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    best_qwk = -1.0
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_qwk": [], "val_acc": []}

    fold_str = f" [Fold {fold}]" if fold is not None else ""

    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model, ordinal_head, train_loader, optimizer, config, loss_type
        )
        val_metrics = evaluate(model, ordinal_head, val_loader, config, loss_type)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_qwk"].append(val_metrics["qwk"])
        history["val_acc"].append(val_metrics["accuracy"])

        if verbose and (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1:3d}{fold_str} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val QWK: {val_metrics['qwk']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

        # Early stopping on QWK
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            patience_counter = 0
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "ordinal_head": {k: v.cpu().clone() for k, v in ordinal_head.state_dict().items()},
                "metrics": val_metrics,
                "epoch": epoch + 1,
            }
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}{fold_str}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        ordinal_head.load_state_dict(best_state["ordinal_head"])

    return model, ordinal_head, best_state["metrics"], history


# ---------- Cross-Validation ----------


def run_cross_validation(dataset, config, embed_dim=256, loss_type="coral"):
    """Run k-fold cross-validation.

    Returns:
        fold_metrics: list of metric dicts per fold
        avg_metrics: averaged metrics across folds
    """
    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )

    # Get labels for stratification
    all_labels = np.array([dataset[i][1].item() for i in range(len(dataset))])

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), all_labels)):
        print(f"\n--- Fold {fold+1}/{config.n_folds} ---")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        _, _, metrics, _ = train_model(
            train_subset, val_subset, config,
            embed_dim=embed_dim, loss_type=loss_type,
            fold=fold + 1,
        )
        fold_metrics.append(metrics)

    # Average metrics
    scalar_keys = ["accuracy", "qwk", "mae", "off_by_1", "off_by_2", "loss"]
    if "ece" in fold_metrics[0]:
        scalar_keys.append("ece")

    avg_metrics = {}
    for key in scalar_keys:
        vals = [m[key] for m in fold_metrics]
        avg_metrics[f"{key}_mean"] = float(np.mean(vals))
        avg_metrics[f"{key}_std"] = float(np.std(vals))

    print(f"\n=== Cross-Validation Results ({config.n_folds} folds) ===")
    for key in scalar_keys:
        print(f"  {key}: {avg_metrics[f'{key}_mean']:.4f} ± {avg_metrics[f'{key}_std']:.4f}")

    return fold_metrics, avg_metrics


# ---------- Main ----------


def parse_args():
    parser = argparse.ArgumentParser(description="Train unimodal Alzheimer's staging model")
    parser.add_argument("--modality", type=str, default="mri", choices=["mri", "audio"])
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--embeddings_path", type=str, default=None)
    parser.add_argument("--labels_path", type=str, default=None)
    parser.add_argument("--loss_type", type=str, default="coral", choices=["coral", "ce"])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--cross_validate", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    # Override config with CLI args
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.fusion_dropout = args.dropout
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.early_stopping_patience = args.patience
    config.n_folds = args.n_folds
    config.seed = args.seed
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.ensure_dirs()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"Device: {config.device}")
    print(f"Modality: {args.modality}")
    print(f"Loss type: {args.loss_type}")
    print(f"Data fraction: {args.data_fraction}")

    # Load data
    if args.synthetic:
        print("Using synthetic data for pipeline testing...")
        dataset = SyntheticEmbeddingDataset(
            n_samples=1000, embed_dim=args.embed_dim, num_classes=config.num_classes
        )
    else:
        emb_path = args.embeddings_path or str(
            config.embedding_dir / f"{args.modality}_embeddings.npz"
        )
        lbl_path = args.labels_path or str(config.embedding_dir / "labels.csv")
        print(f"Loading embeddings: {emb_path}")
        print(f"Loading labels: {lbl_path}")
        dataset = EmbeddingDataset(emb_path, lbl_path, modality=args.modality)

    # Apply data fraction
    if args.data_fraction < 1.0:
        n_use = max(1, int(len(dataset) * args.data_fraction))
        indices = np.random.RandomState(config.seed).permutation(len(dataset))[:n_use]
        dataset = Subset(dataset, indices)
        print(f"Using {n_use}/{len(dataset)} samples ({args.data_fraction*100:.0f}%)")

    if args.cross_validate:
        fold_metrics, avg_metrics = run_cross_validation(
            dataset, config, embed_dim=args.embed_dim, loss_type=args.loss_type
        )

        # Save results
        results_path = config.results_dir / f"unimodal_{args.modality}_{args.loss_type}_cv.csv"
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(avg_metrics.keys()))
            writer.writeheader()
            writer.writerow(avg_metrics)
        print(f"Results saved to {results_path}")
    else:
        # Simple train/val split (80/20)
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        indices = np.random.RandomState(config.seed).permutation(n_total)
        train_subset = Subset(dataset, indices[:n_train])
        val_subset = Subset(dataset, indices[n_train:])

        print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

        model, ordinal_head, best_metrics, history = train_model(
            train_subset, val_subset, config,
            embed_dim=args.embed_dim, loss_type=args.loss_type,
        )

        print(f"\n=== Best Validation Results ===")
        for key in ["accuracy", "qwk", "mae", "off_by_1", "loss"]:
            print(f"  {key}: {best_metrics[key]:.4f}")

        # Save checkpoint
        if args.save_path:
            save_path = args.save_path
        else:
            save_path = str(
                config.checkpoint_dir / f"unimodal_{args.modality}_{args.loss_type}.pt"
            )
        torch.save({
            "model_state": model.state_dict(),
            "ordinal_head_state": ordinal_head.state_dict(),
            "config": {
                "embed_dim": args.embed_dim,
                "loss_type": args.loss_type,
                "modality": args.modality,
            },
            "metrics": {k: v for k, v in best_metrics.items() if k != "confusion_matrix" and k != "severity_scores"},
        }, save_path)
        print(f"Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
