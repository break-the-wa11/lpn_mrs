"""
This files provides a function to generate samples from trained models
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation.sample import compute_stat, fid, GMM_sample, LPN_sample, LPN_cond_sample, GLOW_sample

def eval_sample(
    model,
    true_dataset,
    benchmark_dataset,
    sample_param,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    savestr: str = "savings",
    logger=None,
):
    """
    Evaluate generated samples

    Args:
        true_dataset: dataset for true distribution
        benchmark_dataset: dataset for benchmark distribution
        sample_param: dict containing model-specific parameters for sampling.
        device: Device to run the evaluation on.
        savestr: Directory to save the perturbed results.
        logger: Logger for logging evaluation progress.
    """
    sample_path = os.path.join(savestr, 'sample.npy')

    true_sample = np.array([true_dataset[i] for i in range(len(true_dataset))])
    bm_sample = np.array([benchmark_dataset[i] for i in range(len(benchmark_dataset))])

    os.makedirs(savestr, exist_ok=True)
    model.eval()
    model = model.to(device)

    if sample_param['model_name'] == 'GLOW':
        sample = GLOW_sample(n_samples, model)
    elif sample_param['model_name'] == 'LPN':
        sample = LPN_sample(data=true_sample,
                            model=model,
                            device=device,
                            savestr=savestr,
                            n_samples=sample_param['n_samples'],
                            sigma=sample_param['noise'],
                            max_iter=sample_param['max_iter'])
    elif sample_param['model_name'] == 'LPN_cond' or sample_param['model_name'] == 'LPN_cond_encode_nn':
        sample = LPN_cond_sample(data=true_sample,
                                 model=model,
                                 device=device,
                                 savestr=savestr,
                                 n_samples=sample_param['n_samples'],
                                 sigma_min=sample_param['noise_min'],
                                 sigma_max=sample_param['noise_max'],
                                 max_iter=sample_param['max_iter'])
    else:
        raise ValueError(f"Unknown model type!")
    
    np.save(sample_path, sample)
    plot(sample, sample_param['model_name'], savestr)

    sample = np.load(sample_path)
    n_samples = sample.shape[0]

    gmm_sample = GMM_sample(true_sample, n_samples)

    mu_sample, sigma_sample = compute_stat(sample)
    mu_true, sigma_true = compute_stat(true_sample)
    mu_bm, sigma_bm = compute_stat(bm_sample)
    mu_gmm, sigma_gmm = compute_stat(gmm_sample)

    fid_sample = fid(mu_sample, sigma_sample, mu_true, sigma_true)
    if logger is not None:
        logger.info(f"{sample_param['model_name']} fid: {fid_sample}")

    fid_bm = fid(mu_bm, sigma_bm, mu_true, sigma_true)
    if logger is not None:
        logger.info(f"benchmark fid: {fid_bm}")

    fid_gmm = fid(mu_gmm, sigma_gmm, mu_true, sigma_true)
    if logger is not None:
        logger.info(f"gmm fid: {fid_gmm}")

    plot(true_sample, model_name='True', savestr=savestr)
    plot(gmm_sample, model_name='GMM', savestr=savestr)


def plot(sample, model_name, savestr):
    plot_num = min(10, sample.shape[0])
    plt.figure(figsize=(12, 6))
    for i in range(plot_num):
        plt.plot(sample[i])
    plt.title(f"{model_name} Samples")
    plt.ylim(-0.5,1.5)
    plt.grid(True)
    plt.savefig(os.path.join(savestr, f'samples_{model_name}.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(sample[-1])
    plt.title(f"Example {model_name} Sample")
    plt.ylim(-0.5,1.5)
    plt.grid(True)
    plt.savefig(os.path.join(savestr, f'sample_eg_{model_name}.png'))
    plt.close()