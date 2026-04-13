"""
trainer.py — Training and validation loop.

The Trainer class handles:
1. Training the model for multiple epochs
2. Validating after each epoch to monitor overfitting
3. Saving the best model (based on validation loss)
4. Tracking and printing metrics (loss, accuracy)
"""

import os
import torch
import torch.nn as nn
import numpy as np

import config


class Trainer:
    """
    Handles the full training pipeline.

    Usage:
        trainer = Trainer(model, train_loader, val_loader)
        history = trainer.train(num_epochs=30)
    """

    def __init__(self, model, train_loader, val_loader, device=None,
                 learning_rate=None, weight_decay=None, checkpoint_path=None):
        """
        Args:
            model: the CNN model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: "cuda", "cpu", or None for auto-detect
            learning_rate: learning rate for Adam optimizer
            weight_decay: L2 regularization strength
            checkpoint_path: custom filename for the best model checkpoint
                             (default: "best_model.pt")
        """
        self.checkpoint_path = checkpoint_path or "best_model.pt"
        # Auto-detect GPU
        if device is None:
            if config.DEVICE == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(config.DEVICE)
        else:
            self.device = torch.device(device)

        print(f"Training on: {self.device}")

        # Move model to device (GPU or CPU)
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function: Binary Cross Entropy with Logits
        # "with logits" means it applies sigmoid internally — more numerically stable
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer: Adam (adaptive learning rate, works well out of the box)
        lr = learning_rate or config.LEARNING_RATE
        wd = weight_decay or config.WEIGHT_DECAY
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Learning rate scheduler: reduce LR when validation loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_one_epoch(self):
        """
        Train the model for one full pass through the training data.

        Returns:
            avg_loss: average loss over all batches
            accuracy: fraction of correct predictions
        """
        self.model.train()  # Set model to training mode (enables dropout, batchnorm)

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_features, batch_labels in self.train_loader:
            # Move data to device
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Forward pass: compute predictions
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_labels)

            # Backward pass: compute gradients
            self.optimizer.zero_grad()  # clear old gradients
            loss.backward()             # compute new gradients
            self.optimizer.step()       # update model weights

            # Track metrics
            total_loss += loss.item() * batch_features.size(0)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()  # Disable gradient computation for validation (saves memory)
    def validate(self):
        """
        Evaluate the model on the validation set.

        Returns:
            avg_loss: average validation loss
            accuracy: fraction of correct predictions
        """
        self.model.eval()  # Set model to evaluation mode (disables dropout)

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_features, batch_labels in self.val_loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_labels)

            total_loss += loss.item() * batch_features.size(0)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, num_epochs=None, save_best=True):
        """
        Full training loop: train for multiple epochs, validate each time.

        Args:
            num_epochs: number of epochs (default: from config)
            save_best: whether to save the best model checkpoint

        Returns:
            history: dict with lists of metrics per epoch
                     {"train_loss": [...], "val_loss": [...],
                      "train_acc": [...], "val_acc": [...]}
        """
        if num_epochs is None:
            num_epochs = config.NUM_EPOCHS

        # History tracks metrics for plotting later
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        best_val_loss = float("inf")

        print(f"\nStarting training for {num_epochs} epochs...")
        print("-" * 70)

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_one_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

            # Save history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Print progress
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            )

            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
                path = os.path.join(config.CHECKPOINT_DIR, self.checkpoint_path)
                torch.save(self.model.state_dict(), path)
                print(f"  → Saved best model (val_loss={val_loss:.4f})")

        print("-" * 70)
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")

        return history
