"""
Train GLOW for density estimation
"""

import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def glow_training(
    model,
    train_dataloader,
    val_dataloader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 40000,
    validate_every_n_steps: int = 1000,
    lr: float = 1e-4,
    savestr: str = "weights",
    logger=None,
):
    """
    Train GLOW for density estimation

    Args:
        model: GLOW model
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        device: Device to run the training on.
        num_steps: Total number of training iterations.
        optimizer: Optimizer to use.
        validate_every_n_steps: Frequency of validation.
        lr: Learning rate for the optimizer.
        savestr: Directory to save the model checkpoints.
        logger: Logger for logging training progress.
    """
    os.makedirs(savestr, exist_ok=True)

    model = model.to(device)


    # Initialize the optimizer
    optimizer = torch.optim.Adamax(model.parameters(), lr = lr, weight_decay=1e-5)

    validator = Validator(val_dataloader, logger=logger)

    global_step = 0
    progress_bar = tqdm(total=num_steps, dynamic_ncols=True)
    progress_bar.set_description(f"Train")

    loss_monitor = []
    best_val_loss = float("inf")

    while True:
        for step, batch in enumerate(train_dataloader):
            model.train()

            batch = torch.tensor(batch).unsqueeze(1).to(device)

            loss = model.forward_kld(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().item()

            logs = {
                "loss": train_loss,
            }
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

            if logger is not None:
                logger.info(f"Step {global_step}: loss: {train_loss}")

            if validate_every_n_steps > 0 and (global_step+1) % validate_every_n_steps == 0:
                val_loss = validator.validate(model, global_step, loss_monitor)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(savestr, f"GLOW_best.pt"),
                    )

            global_step += 1
            if global_step >= num_steps:
                break

        if global_step >= num_steps:
            break

    print(f"Training done. Best val loss: {best_val_loss}")
    if logger is not None:
        logger.info(f"Training done. Best val loss: {best_val_loss}")

    df = pd.DataFrame(loss_monitor)
    df.to_csv(os.path.join(savestr, "GLOW_loss.csv"), index=False)
    plot_loss(loss_monitor, savestr)


def plot_loss(loss_monitor, savestr):
    steps = [entry['step'] for entry in loss_monitor]
    loss = [entry['loss'] for entry in loss_monitor]
    plt.figure(figsize=(7,6))
    plt.plot(steps, loss, marker='o', label='Loss', color='b')
    plt.title("Validation Loss over Steps")
    plt.xlabel('Steps')
    plt.ylabel('Validation Loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(savestr, "GLOW_loss.png"), dpi=300, bbox_inches='tight')

    
class Validator:
    """Class for validation."""

    def __init__(self, dataloader, logger=None):
        self.dataloader = dataloader
        self.logger = logger

    def _validate(self, model):
        """Validate the model on the validation dataset."""
        model.eval()
        device = next(model.parameters()).device

        batch = next(iter(self.dataloader))
        batch = torch.tensor(batch).unsqueeze(1).to(device)

        self.loss = model.forward_kld(batch).item()

    def _log(self, step, loss_monitor):
        if self.logger is not None:
            self.logger.info(f"Validation at step {step}: loss: {self.loss}")
        loss_monitor.append({
            "step": step,
            "loss": self.loss,
        })

    def validate(self, model, step, loss_monitor):
        """Validate the model and log the metrics."""

        self._validate(model)
        print(f"Validation at step {step}: loss: {self.loss}")
        self._log(step, loss_monitor)
        return self.loss