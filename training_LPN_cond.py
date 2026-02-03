"""
Train Conditional LPN by proximal matching for denoising 1D signals.
"""

import argparse
import os

import torch
from datasets import MRSDataset
from training_methods.lpn_cond_training import lpn_cond_training
import logging
import datetime
from networks import LPN_cond
from hyperparameters import get_LPN_hyperparameters

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default="data/",
    help="Root directory of dataset.",
)
parser.add_argument(
    "--kernel", type=int, default=101, help="Kernel size for LPN layer."
)
parser.add_argument(
    "--hidden", type=int, default=128, help="Hidden dim for LPN layer."
)
parser.add_argument(
    "--noise_min", type=float, default=0.001, help="Min Noise level for training"
)
parser.add_argument(
    "--noise_max", type=float, default=0.03, help="Max Noise level for training"
)
parser.add_argument(
    "--noise_val", type=float, default=0.01, help="Noise level for validation"
)
parser.add_argument("--batch_size", type=int, default=None)
args = parser.parse_args()

###############################################################################
savestr = f"weights/lpn_cond_mrs_h_{args.hidden}_k_{args.kernel}_n_({args.noise_min}_{args.noise_max})"
if not os.path.isdir("weights"):
    os.mkdir("weights")
if not os.path.isdir(savestr):
    os.mkdir(savestr)

kernel = args.kernel
hidden = args.hidden
noise_min = args.noise_min
noise_max = args.noise_max
noise_val = args.noise_val
batch_size = 64 if args.batch_size is None else args.batch_size
hyper_params = get_LPN_hyperparameters()

###############################################################################
# Create dataset and dataloaders
train_dataset = MRSDataset(args.data_dir, split="train")
val_dataset = MRSDataset(args.data_dir, split="validate")
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size = len(val_dataset),
    shuffle = False,
    drop_last = True,
    num_workers = 8,
)

###############################################################################

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"{savestr}/log_training_lpn_cond_mrs_h_{args.hidden}_k_{args.kernel}_n_({args.noise_min}_{args.noise_max})_" + str(datetime.datetime.now()) + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

###############################################################################
# Training
###############################################################################

model = LPN_cond(
    in_dim=1,
    hidden_c=1,
    hidden=hidden,
    kernel=kernel,
    beta=10,
    alpha=1e-6
).to(device)

lpn_cond_training(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device=device,
    sigma_noise= (noise_min, noise_max),
    sigma_val = noise_val,
    num_steps=hyper_params.num_steps,
    validate_every_n_steps=hyper_params.validate_every_n_steps,
    num_steps_pretrain=hyper_params.num_steps_pretrain,
    num_stages=hyper_params.num_stages,
    gamma_init=hyper_params.gamma_init,
    pretrain_lr=hyper_params.pretrain_lr,
    lr=hyper_params.lr,
    savestr=savestr,
    loss_type="pm",
    logger=logger,
)
logger.info("Training finished.")