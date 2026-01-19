"""
Evaluate sample for models
"""

import argparse
import datetime
import logging
import os
import torch

from datasets import MRSDataset
from networks import LPN, GLOW
from evaluation import eval_sample

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
    help="Root directory of true dataset."
)
parser.add_argument(
    "--model_name",
    type=str,
    default="LPN"
)
parser.add_argument(
    "--kernel", type=int, default=101, help="Kernel size for LPN layer."
)
parser.add_argument(
    "--noise_level", type=float, default=0.01, help="Noise level for training"
)
parser.add_argument(
    "--n_samples", 
    type=int, 
    default=30,
    help="Number of samples for prior evaluation"
)
args = parser.parse_args()

###############################################################################
data_dir = args.data_dir
model_name = args.model_name
n_samples = args.n_samples

if model_name == "LPN":
    kernel = args.kernel
    noise_level = args.noise_level
    savestr = f"savings/lpn_mrs_kernel_{args.kernel}_noise_{args.noise_level}"
    model = LPN(
        in_dim=1,
        hidden=128,
        kernel=kernel,
        beta=10,
        alpha=1e-6
    )
    model.load_state_dict(torch.load(f"weights/lpn_mrs_kernel_{args.kernel}_noise_{args.noise_level}/LPN_best.pt"))
elif model_name == "GLOW":
    savestr = f"savings/glow_mrs"
    model = GLOW(
        L=3,
        K=16,
        input_shape=[1,512],
        hidden_channels=32,
        split_mode='channel',
        scale=True,
    )
    model.load_state_dict(torch.load(f"weights/glow_mrs/GLOW_best.pt"))
else:
    raise ValueError("Unknown model!")

if not os.path.isdir("savings"):
    os.mkdir("savings")
if not os.path.isdir(savestr):
    os.mkdir(savestr)

###############################################################################

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"{savestr}/sample_" + str(datetime.datetime.now()) + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

###############################################################################
# Create dataset
true_dataset = MRSDataset(data_dir, split="train")
benchmark_dataset = MRSDataset(data_dir, split="validate")

###############################################################################
# Evaluation
###############################################################################

eval_sample(
    model=model,
    true_dataset=true_dataset,
    benchmark_dataset=benchmark_dataset,
    model_type=model_name,
    device=device,
    n_samples=n_samples,
    savestr=savestr,
    logger=logger
)