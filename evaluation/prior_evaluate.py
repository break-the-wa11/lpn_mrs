"""
This files provides a function to evaluate prior from trained models
"""

import os
import torch
import numpy as np
import pandas as pd

from evaluation.prior import perturb_generator, eval_lpn_prior
from hyperparameters import get_perturb_hyperparameters

def eval_prior(
    model,
    model_type: str = "GLOW",
    inv_alg: str = 'cvx_cg',
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    generate_data: bool = True,
    n_samples: int = 30,
    perturb_data_dir: str = "data/perturb",
    raw_data_dir: str = "data/",
    savestr: str = "savings",
    logger=None,
):
    """
    Evaluate prior on perturbed samples

    Args:
        model_type: "LPN" or "GLOW"
        inv_alg: Inversion algorithm only for LPN, choose from ['ls', 'cvx_cg', 'cvx_gd']
        device: Device to run the evaluation on.
        generate_data: Whether to generate the perturbed data.
        n_samples: number of samples to evaluate.
        perturb_data_dir: Directory with perturbed data.
        raw_data_dir: str, directory of raw test_data.
        savestr: Directory to save the perturbed results.
        logger: Logger for logging evaluation progress.
    """
    if generate_data or not os.path.isdir(perturb_data_dir):
        os.makedirs(perturb_data_dir, exist_ok=True)
        hyp_list = get_perturb_hyperparameters()
        for hyp in hyp_list:
            perturb_generator(n_samples, hyp, raw_data_dir, perturb_data_dir)

    model.eval()
    model = model.to(device)
    files = [f for f in os.listdir(perturb_data_dir) if f.endswith('.npy')]

    p_list = []
    perturb_list = []

    for file in files:
        data_path = os.path.join(perturb_data_dir, file)
        batch_perturb = np.load(data_path)
        batch_perturb = torch.tensor(batch_perturb).unsqueeze(1).to(device)
        perturb_name = file[:-4]
        perturb_list.append(perturb_name)

        if model_type == "GLOW":
            p = model.log_prob(batch_perturb).detach().cpu().numpy()
        elif model_type == "LPN":
            p, y, fy = eval_lpn_prior(batch_perturb, model, inv_alg=inv_alg)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if logger is not None:
            logger.info(
                f"{perturb_name} prior:"
            )
            logger.info(
                p
            )

        p_list.append(p)

    df = pd.DataFrame(np.array(p_list).T, columns=perturb_list)
    os.makedirs(savestr, exist_ok=True)
    df.to_csv(f"{savestr}/prior.csv", index=False)