"""
Train GLOW for density estimation and sampling.
"""

import argparse
import os

import torch
from datasets import MRSDataset
from training_methods.glow_training import glow_training
import logging
import datetime
from networks import GLOW
from hyperparameters import get_GLOW_hyperparameters

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
parser.add_argument("--batch_size", type=int, default=None)
args = parser.parse_args()

###############################################################################
savestr = f"weights/glow_mrs_5"
if not os.path.isdir("weights"):
    os.mkdir("weights")
if not os.path.isdir(savestr):
    os.mkdir(savestr)

batch_size = 64 if args.batch_size is None else args.batch_size
hyper_params = get_GLOW_hyperparameters()

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
    batch_size = 50,
    shuffle = False,
    drop_last = True,
    num_workers = 8,
)

###############################################################################

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"{savestr}/log_training_glow_mrs_" + str(datetime.datetime.now()) + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

###############################################################################
# Training
###############################################################################

model = GLOW(
    L=3,
    K=16,
    input_shape=[1,512],
    hidden_channels=32,
    split_mode='channel',
    scale=True,
).to(device)

glow_training(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device=device,
    num_steps=hyper_params.num_steps,
    validate_every_n_steps=hyper_params.validate_every_n_steps,
    lr=hyper_params.lr,
    savestr=savestr,
    logger=logger,
)
logger.info("Training finished.")