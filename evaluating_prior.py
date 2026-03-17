"""
Evaluate prior for models
"""

import argparse
import datetime
import logging
import os
import torch

from networks import LPN, LPN_cond, LPN_cond_encode_nn, GLOW
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
    "--kernel", type=int, default=3, help="Kernel size for LPN layer."
)
parser.add_argument(
    "--hidden", type=int, default=32, help="Hidden dim for LPN layer."
)
parser.add_argument(
    "--noise_min", type=float, default=0.05, help="Min noise level for training"
)
parser.add_argument(
    "--noise_max", type=float, default=0.2, help="Max noise level for training"
)
parser.add_argument(
    "--n_samples", 
    type=int, 
    default=30,
    help="Number of samples for prior evaluation"
)
parser.add_argument(
    "--data_type", 
    type=str, 
    default='low_lipid',
    help="Type of data to add gaussian perturbation"
)
args = parser.parse_args()

###############################################################################
data_dir = args.data_dir
model_name = args.model_name
n_samples = args.n_samples
data_type = args.data_type

if model_name == "LPN":
    kernel = args.kernel
    hidden = args.hidden
    noise_min = args.noise_min
    noise_max = args.noise_max
    assert noise_min == noise_max, f'Only one noise level is required for training LPN'
    savestr = f"savings/lpn_mrs_h_{args.hidden}_k_{args.kernel}_n_{noise_min}"
    model = LPN(
        in_dim=1,
        hidden=hidden,
        kernel=kernel,
        beta=10,
        alpha=1e-6
    )
    model.load_state_dict(torch.load(f"weights/lpn_mrs_h_{args.hidden}_k_{args.kernel}_n_{noise_min}/LPN_best.pt"))
    model = model.to(device)
    model_param = {'model_name': model_name,
                   'noise': noise_min}
elif model_name == "LPN_cond":
    kernel = args.kernel
    hidden = args.hidden
    noise_min = args.noise_min
    noise_max = args.noise_max
    savestr = f"savings/lpn_cond_mrs_h_{args.hidden}_k_{args.kernel}_n_({noise_min}_{noise_max})"
    model = LPN_cond(
        in_dim=1,
        hidden_c=1,
        hidden=hidden,
        kernel=kernel,
        beta=10,
        alpha=1e-6
    )
    model.load_state_dict(torch.load(f"weights/lpn_cond_mrs_h_{args.hidden}_k_{args.kernel}_n_({noise_min}_{noise_max})/LPN_best.pt"))
    model = model.to(device)
    model_param = {'model_name': model_name,
                   'noise_min': noise_min,
                   'noise_max': noise_max}
elif model_name == "LPN_cond_encode_nn":
    kernel = args.kernel
    hidden = args.hidden
    noise_min = args.noise_min
    noise_max = args.noise_max
    savestr = f"savings/lpn_cond_encode_nn_mrs_h_{args.hidden}_k_{args.kernel}_n_({args.noise_min}_{args.noise_max})"
    model = LPN_cond_encode_nn(
        in_dim=1,
        hidden_c=1,
        hidden=hidden,
        kernel=kernel,
        beta=10,
        alpha=1e-6
    )
    model.load_state_dict(torch.load(f"weights/lpn_cond_encode_nn_mrs_h_{args.hidden}_k_{args.kernel}_n_({args.noise_min}_{args.noise_max})/LPN_best.pt"))
    model = model.to(device)
    model_param = {'model_name': model_name,
                   'noise_min': noise_min,
                   'noise_max': noise_max}
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
    model = model.to(device)
    model_param = {'model_name': model_name}
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
    model_param=model_param,
    device=device,
    n_samples=n_samples,
    data_type= data_type,
    raw_data_dir=data_dir,
    savestr=f"{savestr}/prior",
    logger=logger
)
logger.info("Evaluation finished.")