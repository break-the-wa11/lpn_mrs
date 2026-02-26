"""
Train LPN by proximal matching
"""

import argparse
from math import dist
import os

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def lpn_cond_training(
    model,
    train_dataloader,
    val_dataloader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 40000,
    optimizer: str = "adam",
    sigma_noise: tuple = (0.04, 0.13),
    validate_every_n_steps: int = 1000,
    validate_noise: list = [0.04, 0.1, 0.13],
    num_steps_pretrain: int = 20000,
    num_stages: int = 4,
    gamma_init: float = 0.1,
    gamma_fix: bool = True,
    pretrain_lr: float = 1e-3,
    lr: float = 1e-4,
    savestr: str = "weights",
    loss_type: str = "pm",
    logger=None,
):
    """
    Train a Conditional LPN using proximal matching.

    Args:
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        device: Device to run the training on.
        num_steps: Total number of training iterations.
        optimizer: Optimizer to use.
        sigma_noise: noise std in prox matching.
        validate_every_n_steps: Frequency of validation.
        num_steps_pretrain: Number of iterations for L1 loss pretraining.
        num_stages: Number of stages in prox matching loss schedule.
        gamma_init: Initial gamma for proximal matching.
        gamma_fix: Whether to fix gamma or scale it with noise level.
        pretrain_lr: Learning rate for L1 loss pretraining.
        lr: Learning rate for prox matching.
        savestr: Directory to save the model checkpoints.
        loss_type: Loss function. ["l2", "l1", "pm"]
            l2: L2 loss always
            l1: L1 loss always
            pm: L1 loss for pretraining, then prox matching loss
        logger: Logger for logging training progress.
    """
    os.makedirs(savestr, exist_ok=True)

    model = model.to(device)

    # Initialize the optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters())

    global_step = 0
    progress_bar = tqdm(total=num_steps, dynamic_ncols=True)
    progress_bar.set_description(f"Train")

    # Loss monitor is to record the loss and later on plot the validation loss
    loss_monitor = []
    best_val_loss_mse_prop = float('inf')

    # Loss parameter setup
    if loss_type == "l2":
        loss_hparams, lr = {"type": "l2"}, lr
    elif loss_type == "l1":
        loss_hparams, lr = {"type": "l1"}, lr
    elif loss_type == "pm":
        lr_init = lr
        if num_steps_pretrain > 0:
            loss_hparams, lr = {"type": "l1"}, pretrain_lr
        else:
            loss_hparams, lr = {"type": "prox_matching", "gamma": gamma_init}, lr

        num_steps_per_stage = (num_steps - num_steps_pretrain) // num_stages
        stage_transition_steps = [num_steps_pretrain + i * num_steps_per_stage for i in range(1, num_stages)] 
    else:
        raise NotImplementedError

    while True:
        for step, batch in enumerate(train_dataloader):
            model.train()

            # set learning rate
            for g in optimizer.param_groups:
                g["lr"] = lr

            # Train step
            loss = train_step(model, optimizer, batch, loss_hparams, sigma_noise, gamma_fix, device)

            logs = {
                "loss": loss,
                "hparams": loss_hparams,
                "lr": lr,
            }
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

            # Update hyperparameters for the proximal matching training during the process
            if loss_type == "pm" and global_step >= num_steps_pretrain:
                # Transit l1 pretraining into proximal matching training
                if global_step == num_steps_pretrain:
                    loss_hparams, lr = {"type": "prox_matching", "gamma": gamma_init}, lr_init
                    gamma = gamma_init
                    stage_loss_min = float('inf')
                
                # After each proximal matching training stage, reduce gamma by a half
                if global_step in stage_transition_steps:
                    lr = lr_init
                    gamma /= 2
                    stage_loss_min = float('inf')
                    loss_hparams["gamma"] = gamma

                # Sometimes the training gets unstable, so if loss increase dramatically, start from previously saved best model and reduce the learning rate by 10
                if loss > stage_loss_min + 0.01 or loss > 1 - 1e-3:
                    lr /= 10
                    model.load_state_dict(
                        torch.load(os.path.join(savestr, f"LPN_best.pt"))
                    )
                
            if logger is not None:
                logger.info(
                    f"Step {global_step}: loss: {loss}, loss_hparams: {loss_hparams}, lr: {lr}"
                )

            # Validate the model and save the best performing model with the lowest validation loss
            if validate_every_n_steps > 0 and (global_step+1) % validate_every_n_steps == 0:
                validator = Validator(val_dataloader, validate_noise, logger)
                val_loss_mse_prop = validator.validate(model, loss_hparams, global_step, loss_monitor, gamma_fix)
                # torch.save(
                #     model.state_dict(),
                #     os.path.join(savestr, f"LPN_step{global_step+1}.pt")
                # )
                if val_loss_mse_prop < best_val_loss_mse_prop:
                    best_val_loss_mse_prop = val_loss_mse_prop
                    torch.save(
                        model.state_dict(),
                        os.path.join(savestr, f"LPN_best.pt")
                    )

            global_step += 1
            if global_step >= num_steps:
                break

            if global_step == num_steps_pretrain:
                print("Pretraining done.")
                torch.save(
                    model.state_dict(),
                    os.path.join(savestr, f"LPN_pretrain.pt"),
                )
            
        if global_step >= num_steps:
            break

    print(f"Training done. Best val MSE loss prop: {best_val_loss_mse_prop}")
    if logger is not None:
        logger.info(f"Training done. Best val MSE loss prop: {best_val_loss_mse_prop}")  

    # Save and plot the loss
    df = pd.DataFrame(loss_monitor)
    df.to_csv(f"{savestr}/LPN_loss.csv", index=False)
    plot_loss(loss_monitor, savestr)


# One training step
def train_step(model, optimizer, batch, loss_hparams, sigma_noise, gamma_fix, device):
    # Prepare ground truth, clean signal
    target = torch.tensor(batch).unsqueeze(1).to(device)
    
    # Prepare noise added, later times by noise_levels
    noise = torch.randn_like(target)

    # prepare noise levels as input to the model, for each element in the batch, the noise std is log uniformly sampled from interval (sigma[0], sigma[1])
    # log_sigma_min = np.log(sigma_noise[0]) 
    # log_sigma_max = np.log(sigma_noise[1])
    # log_uniform = np.random.uniform(log_sigma_min, log_sigma_max, size=target.size(0))
    # noise_levels_np = np.exp(log_uniform)
    # noise_levels = torch.tensor(noise_levels_np, dtype=torch.float32).view(-1, 1).to(device)

    noise_levels = torch.empty(target.size(0)).uniform_(sigma_noise[0], sigma_noise[1]).view(-1,1).to(device)
    
    # Prepare model input, the noisy signal
    input = target + noise * noise_levels.view(-1,1,1)
    output = model(input, noise_levels)

    loss_func = get_loss(loss_hparams)

    if loss_hparams["type"] == "prox_matching":
        loss = loss_func(output, target, noise_levels, gamma_fix)
    else:
        loss = loss_func(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.wclip()

    return loss.detach().item()


def plot_loss(loss_monitor, savestr):
    """Plot average loss and MSE for each bin across training steps."""
    
    # Step 1: Group entries with the same bin_name
    grouped_losses = {}
    grouped_mses = {}
    grouped_mses_prop = {}
    
    for entry in loss_monitor:
        bin_name = entry['val_noise']
        if bin_name not in grouped_losses:
            grouped_losses[bin_name] = []
            grouped_mses[bin_name] = []
            grouped_mses_prop[bin_name] = []
        
        grouped_losses[bin_name].append((entry['step'], entry['loss']))
        grouped_mses[bin_name].append((entry['step'], entry['loss_mse']))
        grouped_mses_prop[bin_name].append((entry['step'], entry['loss_mse_prop']))
    
    # Step 2: Create a subplot for loss
    plt.figure(figsize=(21, 6))

    # Subplot for Validation Loss
    plt.subplot(1, 3, 1)
    for bin_name, losses in grouped_losses.items():
        steps, loss_values = zip(*losses)  # Unzip steps and loss values
        plt.plot(steps, loss_values, marker='o', label=bin_name)  # Plot each bin
        
    plt.title("Validation Loss Across Bins Over Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Average Validation Loss")
    plt.grid()
    plt.xticks(rotation=45)
    plt.legend()
    
    # Step 3: Create a subplot for MSE
    plt.subplot(1, 3, 2)
    for bin_name, mses in grouped_mses.items():
        steps, mse_values = zip(*mses)  # Unzip steps and MSE values
        plt.plot(steps, mse_values, marker='o', label=bin_name)  # Plot each bin
        
    plt.title("MSE Across Bins Over Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Average MSE Loss")
    plt.ylim(0,0.002)
    plt.grid()
    plt.xticks(rotation=45)
    plt.legend()

    # Step 4: Create a subplot for MSE Proportion
    plt.subplot(1, 3, 3)
    for bin_name, mses_prop in grouped_mses_prop.items():
        steps, mse_values = zip(*mses_prop)  # Unzip steps and MSE values
        plt.plot(steps, mse_values, marker='o', label=bin_name)  # Plot each bin
        
    plt.title("MSE Proportion Across Bins Over Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Average MSE Proportion Loss")
    plt.ylim(0,1)
    plt.grid()
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{savestr}/LPN_loss.png", dpi=300, bbox_inches="tight")
    plt.show()


# Validator class that takes data from validation dataset and add fixed level of noise
class Validator:
    """Class for validation."""

    def __init__(self, dataloader, val_noise, logger=None):
        self.dataloader = dataloader
        self.logger = logger
        self.val_noise = val_noise
        self.bin_results = {sigma: [] for sigma in val_noise}

    def _validate(self, model, loss_hparams, gamma_fix):
        """Validate the model on the validation dataset."""
        model.eval()
        device = next(model.parameters()).device

        for sigma in self.val_noise:
            for step, batch in enumerate(self.dataloader):
                target = torch.tensor(batch).unsqueeze(1).to(device)
                noise = torch.randn_like(target)

                input = target + noise * sigma
                noise_levels = torch.full((target.size(0),1), sigma).to(device)
                output = model(input, noise_levels).to(device)

                loss_func = get_loss(loss_hparams)

                if loss_hparams["type"] == "prox_matching":
                    loss = loss_func(output, target, noise_levels, gamma_fix).item()
                else:
                    loss = loss_func(output, target).item()
                
                loss_mse = torch.mean((output - target) ** 2).item()
                loss_mse_prop = loss_mse / sigma**2

                self.bin_results[sigma].append({
                    "loss": loss,
                    "loss_mse": loss_mse,
                    "loss_mse_prop": loss_mse_prop
                })
        
    def _log(self, step, loss_monitor, loss_hparams):
        for sigma in self.val_noise:
            avg_loss = np.mean([r['loss'] for r in self.bin_results[sigma]])
            avg_loss_mse = np.mean([r['loss_mse'] for r in self.bin_results[sigma]])
            avg_loss_mse_prop = np.mean([r['loss_mse_prop'] for r in self.bin_results[sigma]])
            if self.logger is not None:
                self.logger.info(f"Step {step}: Noise {sigma:.3f} - Avg Loss: {avg_loss}, Avg MSE: {avg_loss_mse}, Avg MSE Prop: {avg_loss_mse_prop}")

            loss_monitor.append({
                "step": step,
                "val_noise": f"{sigma:.3f}",
                "loss": avg_loss,
                "loss_mse": avg_loss_mse,
                "loss_mse_prop": avg_loss_mse_prop,
                "loss_type": loss_hparams["type"],
            })

        self.loss = np.mean([r['loss'] for bin_results in self.bin_results.values() for r in bin_results])
        self.loss_mse = np.mean([r['loss_mse'] for bin_results in self.bin_results.values() for r in bin_results])
        self.loss_mse_prop = np.mean([r['loss_mse_prop'] for bin_results in self.bin_results.values() for r in bin_results])
        loss_monitor.append({
            "step": step,
            "val_noise": "all",
            "loss": self.loss,
            "loss_mse": self.loss_mse,
            "loss_mse_prop": self.loss_mse_prop,
            "loss_type": loss_hparams["type"],
        })
        if self.logger is not None:
            self.logger.info(f"Step {step}: All noise levels - Avg Loss: {self.loss}, Avg MSE: {self.loss_mse}, Avg MSE Prop: {self.loss_mse_prop}")

    def validate(self, model, loss_hparams, step, loss_monitor, gamma_fix):
        """Validate the model and log the metrics."""

        self._validate(model, loss_hparams, gamma_fix)
        self._log(step, loss_monitor, loss_hparams)

        return self.loss_mse


##########################
# Utils for loss function
##########################
# def get_loss_hparams_and_lr(args, global_step):
#     """Get loss hyperparameters and learning rate based on training schedule.
#     Parameters:
#         args (argparse.Namespace): Arguments from command line.
#         global_step (int): Current training step.
#     """
#     if global_step < args.num_steps_pretrain:
#         loss_hparams, lr = {"type": "l1"}, args.pretrain_lr
#     else:
#         num_steps = args.num_steps - args.num_steps_pretrain
#         step = global_step - args.num_steps_pretrain

#         def _get_loss_hparams_and_lr(num_steps, step):
#             num_steps_per_stage = num_steps // args.num_stages
#             stage = step // num_steps_per_stage
#             if stage >= args.num_stages:
#                 stage = args.num_stages - 1
#             loss_hparams = {
#                 "type": "prox_matching",  # proximal matching
#                 "gamma": args.gamma_init / (2 ** stage),
#             }
#             lr = args.lr
#             return loss_hparams, lr
        
#         loss_hparams, lr = _get_loss_hparams_and_lr(num_steps, step)

#     return loss_hparams, lr


def get_loss(loss_hparams):
    """Get loss function from hyperparameters.
    Parameters:
        loss_hparams (dict): Hyperparameters for loss function.
    """
    if loss_hparams["type"] == "l2":
        return nn.MSELoss()
    elif loss_hparams["type"] == "l1":
        return nn.L1Loss()
    elif loss_hparams["type"] == "prox_matching":
        return ExpDiracSrgt(gamma=loss_hparams["gamma"])
    else:
        raise NotImplementedError
    

# surrogate L0 loss: -exp(-(x/gamma)^2) + 1
def exp_func(x, gamma):
    return -torch.exp(-((x / gamma) ** 2)) + 1


class ExpDiracSrgt(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target, noise_levels, gamma_fix):
        """
        input, target: batch, *
        noise_levels: (batch, 1)
        """
        bsize = input.shape[0]
        dist = (input - target).pow(2).reshape(bsize, -1).mean(1).sqrt()
        
        if gamma_fix:
            return exp_func(dist, self.gamma).mean()
        else:
            return exp_func(dist, self.gamma / 0.1 * noise_levels.squeeze(1)).mean()