"""
Evaluate prior for models
"""

import argparse
import datetime
import logging
import os
import torch

from networks import LPN, GLOW
from evaluation import eval_denoise

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
    help="Root directory of raw dataset.",
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
    "--noise_min", type=float, default=0.001, help="Min noise level during training"
)
parser.add_argument(
    "--noise_max", type=float, default=0.03, help="Max noise level during training"
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
    savestr = f"savings/lpn_mrs_kernel_{args.kernel}_noise_({args.noise_min}_{args.noise_max})"
    model = LPN(
        in_dim=1,
        hidden=128,
        kernel=kernel,
        beta=10,
        alpha=1e-6
    )
    model.load_state_dict(torch.load(f"weights/lpn_mrs_kernel_{args.kernel}_noise_({args.noise_min}_{args.noise_max})/LPN_best.pt"))
elif model_name == "GLOW":
    savestr = f"savings/glow_mrs_1"
    model = GLOW(
        L=3,
        K=16,
        input_shape=[1,512],
        hidden_channels=32,
        split_mode='channel',
        scale=True,
    )
    model.load_state_dict(torch.load(f"weights/glow_mrs_1/GLOW_best.pt"))
else:
    raise ValueError("Unknown model!")

if not os.path.isdir("savings"):
    os.mkdir("savings")
if not os.path.isdir(savestr):
    os.mkdir(savestr)

###############################################################################

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"{savestr}/denoise_" + str(datetime.datetime.now()) + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

###############################################################################
# Evaluation
###############################################################################

eval_denoise(
    model=model,
    model_type=model_name,
    device=device,
    n_samples=n_samples,
    data_dir=data_dir,
    savestr=f"{savestr}/denoise/",
    logger=logger
)
logger.info("Evaluation finished.")