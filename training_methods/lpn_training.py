"""
Train LPN by proximal matching
"""

import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def lpn_training(
    model,
    train_dataloader,
    val_dataloader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 40000,
    optimizer: str = "adam",
    sigma_noise: tuple = (0.001, 0.03),
    sigma_val: float = 0.01,
    validate_every_n_steps: int = 1000,
    num_steps_pretrain: int = 20000,
    num_stages: int = 4,
    gamma_init: float = 0.1,
    pretrain_lr: float = 1e-3,
    lr: float = 1e-4,
    savestr: str = "weights",
    loss_type: str = "pm",
    logger=None,
):
    """
    Train a LPN using proximal matching.

    Args:
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        device: Device to run the training on.
        num_steps: Total number of training iterations.
        optimizer: Optimizer to use.
        sigma_noise: noise std in prox matching.
        sigma_val: noise std in validation.
        validate_every_n_steps: Frequency of validation.
        num_steps_pretrain: Number of iterations for L1 loss pretraining.
        num_stages: Number of stages in prox matching loss schedule.
        gamma_init: Initial gamma for proximal matching.
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

    validator = Validator(val_dataloader, sigma_val, logger)

    global_step = 0
    progress_bar = tqdm(total=num_steps, dynamic_ncols=True)
    progress_bar.set_description(f"Train")

    loss_monitor = []
    best_val_loss_mse = float('inf')

    if loss_type == "l2":
        loss_hparams, lr = {"type": "l2"}, lr
    elif loss_type == "l1":
        loss_hparams, lr = {"type": "l1"}, lr
    elif loss_type == "pm":
        if num_steps_pretrain > 0:
            loss_hparams, lr = {"type": "l1"}, pretrain_lr
        else:
            loss_hparams, lr = {"type": "prox_matching", "gamma": gamma_init}, lr

        num_steps_per_stage = (num_steps - num_steps_pretrain) // num_stages
        stage_transition_steps = [num_steps_pretrain + i * num_steps_per_stage for i in range(1, num_stages)] 

        lr_init = lr
    else:
        raise NotImplementedError

    while True:
        for step, batch in enumerate(train_dataloader):
            model.train()

            # get loss
            loss_func = get_loss(loss_hparams)
            # set learning rate
            for g in optimizer.param_groups:
                g["lr"] = lr

            # Train step
            loss = train_step(model, optimizer, batch, loss_func, sigma_noise, device)

            logs = {
                "loss": loss,
                "hparams": loss_hparams,
                "lr": lr,
            }
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

            if loss_type == "pm" and global_step >= num_steps_pretrain:
                if global_step == num_steps_pretrain:
                    loss_hparams, lr = {"type": "prox_matching", "gamma": gamma_init}, lr_init
                    gamma = gamma_init
                    stage_loss_min = float('inf')
                
                if global_step in stage_transition_steps:
                    lr = lr_init
                    gamma /= 2
                    stage_loss_min = float('inf')
                    loss_hparams["gamma"] = gamma
                if loss > stage_loss_min + 0.01 or loss > 1 - 1e-3:
                    lr /= 10
                    model.load_state_dict(
                        torch.load(os.path.join(savestr, f"LPN_best.pt"))
                    )
                
            if logger is not None:
                logger.info(
                    f"Step {global_step}: loss: {loss}, loss_hparams: {loss_hparams}, lr: {lr}"
                )

            if validate_every_n_steps > 0 and (global_step+1) % validate_every_n_steps == 0:
                val_loss_mse = validator.validate(model, loss_hparams, global_step, loss_monitor)
                # torch.save(
                #     model.state_dict(),
                #     os.path.join(savestr, f"LPN_step{global_step+1}.pt")
                # )
                if val_loss_mse < best_val_loss_mse:
                    best_val_loss_mse = val_loss_mse
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

    print(f"Training done. Best val MSE loss: {best_val_loss_mse}")
    if logger is not None:
        logger.info(f"Training done. Best val MSE loss: {best_val_loss_mse}")  

    df = pd.DataFrame(loss_monitor)
    df.to_csv(f"{savestr}/LPN_loss.csv", index=False)
    plot_loss(loss_monitor, savestr)


def train_step(model, optimizer, batch, loss_func, sigma_noise, device):
    target = torch.tensor(batch).unsqueeze(1).to(device)
    noise = torch.randn_like(target)
    noise_levels = torch.empty(target.size(0)).uniform_(*sigma_noise)
    input = target + noise * noise_levels.view(-1,1,1)
    output = model(input)

    loss = loss_func(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.wclip()

    return loss.detach().item()


def plot_loss(loss_monitor, savestr):
    steps = [entry['step'] for entry in loss_monitor]
    loss = [np.log10(entry['loss']) for entry in loss_monitor]
    mse = [np.log10(entry['loss_mse']) for entry in loss_monitor]
    plt.figure(figsize=(14,6))

    plt.subplot(1,2,1)
    plt.plot(steps, loss, marker='o', label='Loss', color='b')
    plt.title("Log Validation Loss over Steps")
    plt.xlabel('Steps')
    plt.ylabel('Log Validation Loss')
    plt.grid()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(steps, mse, marker='o', label='MSE', color='b')
    plt.title("Log MSE of Denoising over Steps")
    plt.xlabel('Steps')
    plt.ylabel('Log MSE Loss')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{savestr}/LPN_loss.png",
                dpi=300,
                bbox_inches="tight")


class Validator:
    """Class for validation."""

    def __init__(self, dataloader, sigma_noise, logger=None):
        self.dataloader = dataloader
        assert type(sigma_noise) == float
        self.sigma_noise = sigma_noise
        self.logger = logger

    def _validate(self, model, loss_hparams):
        """Validate the model on the validation dataset."""
        model.eval()
        device = next(model.parameters()).device

        batch = next(iter(self.dataloader))
        target = torch.tensor(batch).unsqueeze(1).to(device)
        noise = torch.randn_like(target)
        input = target + noise * self.sigma_noise

        output = model(input)

        loss_func = get_loss(loss_hparams)

        loss = loss_func(output, target).item()
        loss_mse = torch.mean((output - target) ** 2).item()

        self.loss = loss
        self.loss_mse = loss_mse
    
    def _log(self, step, loss_monitor, loss_hparams):
        if self.logger is not None:
            self.logger.info(f"Validation at step {step}: PM loss: {self.loss}, MSE loss: {self.loss_mse}")
        loss_monitor.append({
            "step": step,
            "loss": self.loss,
            "loss_mse": self.loss_mse,
            "loss_type": loss_hparams["type"],
        })

    def validate(self, model, loss_hparams, step, loss_monitor):
        """Validate the model and log the metrics."""

        self._validate(model, loss_hparams)
        print(
            f"Validation at step {step}: PM loss: {self.loss}, MSE loss: {self.loss_mse}"
        )
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

    def forward(self, input, target):
        """
        input, target: batch, *
        """
        bsize = input.shape[0]
        dist = (input - target).pow(2).reshape(bsize, -1).mean(1).sqrt()
        return exp_func(dist, self.gamma).mean()