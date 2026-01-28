"""
This files provides a function to denoise from trained models
"""

import os
import torch
import numpy as np
import pandas as pd
import random

from datasets import MRSDataset
from evaluation.denoise import wv_denoise, lpn_denoise
from evaluation.prior import perturb_generator
from hyperparameters import get_denoise_hyperparameters, get_wv_hyperparameters

def eval_denoise(
    model,
    model_type,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_samples: int = 30,
    data_dir: str = "data/",
    savestr: str = "savings",
    logger=None,
):
    """
    Evaluate prior on perturbed samples

    Args:
        model_type: "LPN" or "GLOW"
        device: Device to run the evaluation on.
        n_samples: number of samples to evaluate.
        data_dir: str, directory of raw test_data.
        savestr: Directory to save the denoised results.
        logger: Logger for logging evaluation progress.
    """
    os.makedirs(savestr, exist_ok=True)

    dataset = MRSDataset(root=data_dir, split='validate', data_type='metab_lipid')
    n_samples = min(n_samples, len(dataset))
    random_indices = random.sample(range(len(dataset)), n_samples)

    gt = np.array([dataset[idx] for idx in random_indices])
    np.save(f"{savestr}/gt.npy", gt)

    model.eval()
    model = model.to(device)

    hyp_list = get_denoise_hyperparameters()
    thresh_dict = get_wv_hyperparameters()

    wv_imp_list = []
    model_imp_list = []
    sigma_list = []

    for hyp in hyp_list:
        if hyp['perturb_mode'] == 'gaussian':
            sigma = hyp['parameters']['sigma']
            x_noisy = gt + np.random.normal(0, sigma, gt.shape)

            thresh = thresh_dict[sigma]

            y_wv = wv_denoise(x_noisy, threshold_factor=thresh)

            if model_type == 'LPN':
                x_tensor = torch.tensor(x_noisy).unsqueeze(1).to(device)
                y_model = lpn_denoise(x_tensor, model)
            elif model_type == 'GLOW':
                pass

            mse = np.mean((x_noisy - gt) ** 2, axis=1)
            mse_model = np.mean((y_model - gt) ** 2, axis=1)
            mse_wv = np.mean((y_wv - gt) ** 2, axis=1)

            wv_improvement = mse_wv / mse
            model_improvement = mse_model / mse

            if logger is not None:
                logger.info(
                    f"Gaussian {sigma} denoise: "
                )
                logger.info(
                    f"wavelet improvement: {wv_improvement}, {model_type} improvement: {model_improvement}"
                )

            wv_imp_list.append(wv_improvement)
            model_imp_list.append(model_improvement)
            sigma_list.append(sigma)

            os.makedirs(f'{savestr}/{sigma}/', exist_ok=True)

            np.save(f'{savestr}/{sigma}/x_noisy.npy', x_noisy)
            np.save(f'{savestr}/{sigma}/y_wv.npy', y_wv)
            np.save(f'{savestr}/{sigma}/y_{model_type}.npy', y_model)

    df_model = pd.DataFrame(np.array(model_imp_list).T, columns=sigma_list)
    df_wv = pd.DataFrame(np.array(wv_imp_list).T, columns=sigma_list)
    df_model.to_csv(f"{savestr}/{model_type}_denoise.csv", index=False)
    df_wv.to_csv(f"{savestr}/wv_denoise.csv", index=False)