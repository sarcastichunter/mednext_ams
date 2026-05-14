import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device="cuda",
        mixed_precision=True,
        save_dir="checkpoints",
        scheduler=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler(enabled=mixed_precision)
        self.scheduler = scheduler

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False)

        for batch in loop:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.mixed_precision):
                preds = self.model(images)
                loss = self.loss_fn(preds, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        if self.scheduler:
            self.scheduler.step()

        return epoch_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        self.model.eval()
        epoch_loss = 0

        loop = tqdm(self.val_loader, desc=f"Epoch {epoch} [val]", leave=False)

        with torch.no_grad():
            for batch in loop:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                with autocast(enabled=self.mixed_precision):
                    preds = self.model(images)
                    loss = self.loss_fn(preds, labels)

                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        return epoch_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss):
        ckpt_path = os.path.join(self.save_dir, f"epoch_{epoch}_loss_{val_loss:.4f}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_state": self.scaler.state_dict(),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

    def fit(self, num_epochs):
        best_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            print(f"[Epoch {epoch}] Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
