"""
This files provides a function to denoise from trained models
"""

import os
import torch
import numpy as np
import pandas as pd
import random

from datasets import MRSDataset
from evaluation.denoise import wv_denoise, lpn_denoise, lpn_cond_denoise
from hyperparameters import get_wv_hyperparameters

def eval_denoise(
    model,
    model_type,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_samples: int = 30,
    data_type: str = "low_lipid", 
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

    dataset_mrs = MRSDataset(root=data_dir, split='test', data_type=data_type)

    model_cols = {}
    wv_cols = {}

    # list sigma folders under data_dir/noise/data_type, parse as float and sort
    sigma_root = os.path.join(data_dir, 'noise', data_type)
    if not os.path.isdir(sigma_root):
        raise FileNotFoundError(f"Noise folder not found: {sigma_root}")

    sigma_names = []
    for name in os.listdir(sigma_root):
        full = os.path.join(sigma_root, name)
        if not os.path.isdir(full):
            continue
        try:
            float(name)  # ensure numeric folder name
            sigma_names.append(name)
        except ValueError:
            # skip non-numeric folder names
            continue

    # sort numerically
    sigma_names = sorted(sigma_names, key=lambda s: float(s))

    for sigma in sigma_names:
        try:
            dataset_noise = MRSDataset(root=data_dir, split='noise', data_type=f"{data_type}/{sigma}")
        except Exception as e:
            if logger is not None:
                logger.warning(f"Skipping sigma {sigma}: cannot create dataset: {e}")
            continue

        n = min(n_samples, len(dataset_mrs))
        if n <= 0:
            if logger is not None:
                logger.warning(f"Skipping sigma {sigma}: no samples available (n={n})")
            continue

        gt = np.array([dataset_mrs[idx] for idx in range(n)])
        noise = np.array([dataset_noise[idx] for idx in range(n)])
        x_noisy = gt + noise

        model.eval()
        model = model.to(device)
        
        thresh_dict = get_wv_hyperparameters()
        if sigma not in thresh_dict:
            if logger is not None:
                logger.warning(f"No threshold for sigma {sigma}, skipping")
            continue
        thresh = thresh_dict[sigma]

        # denoise with wavelet
        try:
            y_wv = wv_denoise(x_noisy, threshold_factor=thresh)
            # ensure y_wv is numpy
            y_wv = np.asarray(y_wv)
        except Exception as e:
            if logger is not None:
                logger.warning(f"Wavelet denoise failed for sigma {sigma}: {e}")
            continue

        try:
            x_tensor = torch.tensor(x_noisy).unsqueeze(1).to(device)
            if model_type == 'LPN':
                y_model_t = lpn_denoise(x_tensor, model)
            elif model_type in ('LPN_cond', 'LPN_cond_encode_nn'):
                b = x_tensor.size(0)
                noise_levels = torch.full((b, 1), float(sigma)).to(device)
                y_model_t = lpn_cond_denoise(x_tensor, model, noise_levels)
            elif model_type == 'GLOW':
                # placeholder: implement as needed
                raise NotImplementedError("GLOW denoiser not implemented here")
            else:
                raise NotImplementedError(f"Unknown model type: {model_type}")

            # ensure numpy arrays for comparisons
            if isinstance(y_model_t, torch.Tensor):
                y_model = y_model_t.detach().cpu().numpy()
            else:
                y_model = np.asarray(y_model_t)

        except Exception as e:
            if logger is not None:
                logger.warning(f"Model denoise failed for sigma {sigma}: {e}")
            continue

        # compute mse arrays
        try:
            mse = np.mean((x_noisy - gt) ** 2, axis=1)
            mse_model = np.mean((y_model - gt) ** 2, axis=1)
            mse_wv = np.mean((y_wv - gt) ** 2, axis=1)
        except Exception as e:
            if logger is not None:
                logger.warning(f"Error computing MSE for sigma {sigma}: {e}")
            continue

        # verify shapes
        if not (mse.shape[0] == mse_model.shape[0] == mse_wv.shape[0] == n):
            if logger is not None:
                logger.warning(f"Shape mismatch for sigma {sigma}: mse {mse.shape}, model {mse_model.shape}, wv {mse_wv.shape}")
            continue

        wv_improvement = mse_wv / mse
        model_improvement = mse_model / mse

        if logger is not None:
            logger.info(
                f"Gaussian {sigma} denoise: "
            )
            logger.info(
                f"wavelet improvement: {wv_improvement}, {model_type} improvement: {model_improvement}"
            )

        # save per-sigma arrays and denoised results
        out_dir = os.path.join(savestr, sigma)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'x_noisy.npy'), x_noisy)
        np.save(os.path.join(out_dir, 'y_wv.npy'), y_wv)
        np.save(os.path.join(out_dir, f'y_{model_type}.npy'), y_model)

        # store columns keyed by sigma (string)
        model_cols[sigma] = model_improvement
        wv_cols[sigma] = wv_improvement

    # After loop, build dataframes if we have any columns
    if model_cols:
        df_model = pd.DataFrame(model_cols)
        df_model.to_csv(os.path.join(savestr, f"{model_type}_denoise.csv"), index=False)
    else:
        if logger is not None:
            logger.warning("No model columns to save")

    if wv_cols:
        df_wv = pd.DataFrame(wv_cols)
        df_wv.to_csv(os.path.join(savestr, "wv_denoise.csv"), index=False)
    else:
        if logger is not None:
            logger.warning("No wavelet columns to save")