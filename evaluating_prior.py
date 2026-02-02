"""
Evaluate prior for models
"""

import argparse
import datetime
import logging
import os
import torch

from networks import LPN, GLOW
from evaluation import eval_prior

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw_data_dir",
    type=str,
    default="data/",
    help="Root directory of raw dataset.",
)
parser.add_argument(
    "--perturb_data_dir",
    type=str,
    default="data/perturb/",
    help="Root directory of perturbed dataset.",
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
    "--hidden", type=int, default=128, help="Hidden dim for LPN layer."
)
parser.add_argument(
    "--noise_min", type=float, default=0.001, help="Min noise level for training"
)
parser.add_argument(
    "--noise_max", type=float, default=0.03, help="Max noise level for training"
)
parser.add_argument(
    "--n_samples", 
    type=int, 
    default=30,
    help="Number of samples for prior evaluation"
)
args = parser.parse_args()

###############################################################################
raw_data_dir = args.raw_data_dir
perturb_data_dir = args.perturb_data_dir
model_name = args.model_name
n_samples = args.n_samples

if model_name == "LPN":
    kernel = args.kernel
    hidden = args.hidden
    savestr = f"savings/lpn_mrs_h_{args.hidden}_k_{args.kernel}_n_({args.noise_min}_{args.noise_max})"
    model = LPN(
        in_dim=1,
        hidden=hidden,
        kernel=kernel,
        beta=10,
        alpha=1e-6
    )
    model.load_state_dict(torch.load(f"weights/lpn_mrs_h_{args.hidden}_k_{args.kernel}_n_({args.noise_min}_{args.noise_max})/LPN_best.pt"))
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
    filename=f"{savestr}/prior_" + str(datetime.datetime.now()) + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

###############################################################################
# Evaluation
###############################################################################

eval_prior(
    model=model,
    model_type=model_name,
    device=device,
    generate_data=False,
    n_samples=n_samples,
    perturb_data_dir=perturb_data_dir,
    raw_data_dir=raw_data_dir,
    savestr=savestr,
    logger=logger
)
logger.info("Evaluation finished.")