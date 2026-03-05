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
    savestr: str = "savings/model_name/denoise/",
    wv_savestr: str = "savings/wv/denoise/",
    osprey_savestr: str = "savings/osprey/denoise/",
    global_savestr: str = "savings/denoise/",
    logger=None,
):
    """
    Evaluate prior on perturbed samples

    Args:
        model_type: "LPN" or "GLOW"
        device: Device to run the evaluation on.
        n_samples: number of samples to evaluate.
        data_dir: str, directory of raw test_data.
        savestr: Directory to save the denoised results of the model.
        wv_savestr: Directory to save the denoised results of wavelet denoising.
        osprey_savestr: Directory to save the denoised results of osprey fitting.
        global_savestr: Directory to save global results like noisy signals.
        logger: Logger for logging evaluation progress.
    """
    os.makedirs(global_savestr, exist_ok=True)
    os.makedirs(savestr, exist_ok=True)
    os.makedirs(wv_savestr, exist_ok=True)
    os.makedirs(osprey_savestr, exist_ok=True)

    dataset_mrs = MRSDataset(root=data_dir, split='test', data_type=data_type)

    n = min(n_samples, len(dataset_mrs))
    if n <= 0:
        if logger is not None:
            logger.warning(f"no samples available (n={n})")

    gt = np.array([dataset_mrs[idx] for idx in range(n)])
    np.save(os.path.join(global_savestr, 'gt.npy'), gt)
    
    model_cols = {}
    wv_cols = {}
    osprey_cols = {}

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
            dataset_osprey = MRSDataset(root=data_dir, split='osprey_denoised', data_type=f"{data_type}/{sigma}")
        except Exception as e:
            if logger is not None:
                logger.warning(f"Skipping sigma {sigma}: cannot create dataset: {e}")
            continue

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

        # denoise with osprey
        y_osprey = np.array([dataset_osprey[idx] for idx in range(n)])

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
            mse_osprey = np.mean((y_osprey - gt) ** 2, axis=1)
        except Exception as e:
            if logger is not None:
                logger.warning(f"Error computing MSE for sigma {sigma}: {e}")
            continue

        # verify shapes
        if not (mse.shape[0] == mse_model.shape[0] == mse_wv.shape[0] == mse_osprey.shape[0] == n):
            if logger is not None:
                logger.warning(f"Shape mismatch for sigma {sigma}: mse {mse.shape}, model {mse_model.shape}, wv {mse_wv.shape}")
            continue

        wv_improvement = mse_wv / mse
        osprey_improvement = mse_osprey / mse
        model_improvement = mse_model / mse

        if logger is not None:
            logger.info(
                f"Gaussian {sigma} denoise: "
            )
            logger.info(
                f"wavelet improvement: {wv_improvement}, osprey improvement: {osprey_improvement}, {model_type} improvement: {model_improvement}"
            )

        # save per-sigma arrays and denoised results
        noisy_dir = os.path.join(global_savestr, sigma)
        wv_dir = os.path.join(wv_savestr, sigma)
        osprey_dir = os.path.join(osprey_savestr, sigma)
        model_dir = os.path.join(savestr, sigma)
        os.makedirs(noisy_dir, exist_ok=True)
        os.makedirs(wv_dir, exist_ok=True)
        os.makedirs(osprey_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        np.save(os.path.join(noisy_dir, 'x_noisy.npy'), x_noisy)
        np.save(os.path.join(wv_dir, 'y_wv.npy'), y_wv)
        np.save(os.path.join(osprey_dir, 'y_osprey.npy'), y_osprey)
        np.save(os.path.join(model_dir, f'y_{model_type}.npy'), y_model)

        # store columns keyed by sigma (string)
        model_cols[sigma] = model_improvement
        wv_cols[sigma] = wv_improvement
        osprey_cols[sigma] = osprey_improvement

    # After loop, build dataframes if we have any columns
    if model_cols:
        df_model = pd.DataFrame(model_cols)
        df_model.to_csv(os.path.join(savestr, f"{model_type}_denoise.csv"), index=False)
    else:
        if logger is not None:
            logger.warning("No model columns to save")

    if wv_cols:
        df_wv = pd.DataFrame(wv_cols)
        df_wv.to_csv(os.path.join(wv_savestr, "wv_denoise.csv"), index=False)
    else:
        if logger is not None:
            logger.warning("No wavelet columns to save")

    if osprey_cols:
        df_osprey = pd.DataFrame(osprey_cols)
        df_osprey.to_csv(os.path.join(osprey_savestr, "osprey_denoise.csv"), index=False)
    else:
        if logger is not None:
            logger.warning("No osprey columns to save")