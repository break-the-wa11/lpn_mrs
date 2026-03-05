"""
This files provides a function to evaluate prior from trained models
"""

import os
import torch
import numpy as np
import pandas as pd

from evaluation.prior import eval_lpn_prior, eval_lpn_cond_prior
from datasets import MRSDataset

def eval_prior(
    model,
    model_param,
    inv_alg: str = 'cvx_cg',
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_samples: int = 30,
    data_type: str = "low_lipid",
    raw_data_dir: str = "data/",
    savestr: str = "savings",
    logger=None,
):
    """
    Evaluate prior on perturbed samples

    Args:
        model_param: dict containing model-specific parameters
        inv_alg: Inversion algorithm only for LPN, choose from ['ls', 'cvx_cg', 'cvx_gd']
        device: Device to run the evaluation on.
        n_samples: number of samples to evaluate.
        data_type: data type to add gaussian noise on.
        raw_data_dir: str, directory of raw test_data.
        savestr: Directory to save the perturbed results.
        logger: Logger for logging evaluation progress.
    """
    model.eval()
    model = model.to(device)
    model_name = model_param['model_name']

    samples = {}

    for perturb_mode in ['clean', 'baseline', 'low_lipid', 'borderline_lipid', 'high_lipid']:
        dataset = MRSDataset(root=raw_data_dir, split='test', data_type=perturb_mode)
        n_samples = min(n_samples, len(dataset))
        samples[perturb_mode] = np.array([dataset[id] for id in range(n_samples)])

    gt = samples[data_type]
    sigma_root = os.path.join(raw_data_dir, 'noise', data_type)
    for sigma in os.listdir(sigma_root):
        dataset_noise = MRSDataset(raw_data_dir, split='noise', data_type=f"{data_type}/{sigma}")
        noise = np.array([dataset_noise[id] for id in range(len(gt))])
        samples[sigma] = gt + noise       

    p_list = []
    for pm in samples.keys():
        sample = samples[pm]
        batch = torch.tensor(sample).unsqueeze(1).to(device)

        if model_name == "GLOW":
            p = model.log_prob(batch).detach().cpu().numpy()
            p_list.append({'pm': pm, 'p': p, 's': None})
            if logger is not None:
                logger.info(f"Mode: {pm} | {model_name} log_prob: {p}")

        elif model_name == "LPN":
            p, y, fy = eval_lpn_prior(batch, model, inv_alg=inv_alg)
            p_list.append({'pm': pm, 'p': p, 's': None})
            if logger is not None:
                logger.info(f"Mode: {pm} | {model_name} log_prob: {p}")
            
        elif model_name == "LPN_cond" or model_name == "LPN_cond_encode_nn":
            sigma_list = np.linspace(model_param["noise_min"], model_param["noise_max"], num=4)
            for s in sigma_list:
                p, y, fy = eval_lpn_cond_prior(batch, model, inv_alg=inv_alg, sigma=s)
                p_list.append({'pm': pm, 'p': p, 's': s})
                if logger is not None:
                    logger.info(f"Mode: {pm} | s: {s} | {model_name} log_prob: {p}")
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    os.makedirs(savestr, exist_ok=True)

    # fixed perturbation order
    fixed_order = ['clean', 'baseline', 'low_lipid', 'borderline_lipid', 'high_lipid']

    unique_sigmas = {item['s'] for item in p_list}
    for s_val in unique_sigmas:
        # Filter p_list for the current sigma group
        current_group = [item for item in p_list if item['s'] == s_val]
        
        # Build dictionary for DataFrame: { 'pm_name': [p_values] }
        csv_data = {item['pm']: item['p'].flatten() for item in current_group}

        # Decide column order:
        cols = [c for c in fixed_order if c in csv_data]
        remaining = [c for c in csv_data.keys() if c not in cols]
        numeric_keys = []
        other_keys = []
        for k in remaining:
            try:
                numeric_keys.append((k, float(k)))
            except Exception:
                other_keys.append(k)
        numeric_keys_sorted = [k for k, _ in sorted(numeric_keys, key=lambda x: x[1])]
        cols.extend(numeric_keys_sorted)
        cols.extend(sorted(other_keys))

        df = pd.DataFrame({c: csv_data[c] for c in cols})

        # Define filename based on model type
        if s_val is None:
            filename = f"prior.csv"
        else:
            # Use f-string format as requested
            filename = f"{s_val:.2f}_prior.csv"
            
        csv_path = os.path.join(savestr, filename)
        df.to_csv(csv_path, index=False)
        
        if logger:
            logger.info(f"Successfully saved {csv_path}")