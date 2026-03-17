"""
This file generate samples for multiple methods
"""

from sklearn.mixture import GaussianMixture
import torch
import numpy as np

def GMM_sample(data, n_samples):
    """Fit GMM model to given data: np array"""
    gmm = GaussianMixture(n_components=1)
    gmm.fit(data)

    gmm_sample = gmm.sample(n_samples)[0]
    return gmm_sample

def LPN_sample(data, 
               n_samples, 
               model, 
               device, 
               savestr,
               noise_schedule,
               cond = True):
    """Sample from lpn using Langevin Dynamics"""
    x = np.mean(data, axis = 0)
    x = torch.tensor(x).unsqueeze(0).unsqueeze(1).repeat(n_samples, 1, 1).to(device)

    sample_all = []
    for it in range(len(noise_schedule)):
        sigma = noise_schedule[it]
        n = torch.randn_like(x) * sigma * np.sqrt(2)
        if cond:
            noise_levels = torch.full((n_samples,1), sigma).to(device)
            x = model(x + n, noise_levels)
        else:
            x = model(x + n)

        sample_all.append(x.squeeze(1).detach().cpu().numpy())

    sample_all = np.array(sample_all)
    np.save(f'{savestr}/sample_all_lpn.npy', sample_all)
    return sample_all[-1]


def GLOW_sample(n_samples, model):
    """Sample from Glow model"""
    with torch.no_grad():
        sample, _ = model.sample(n_samples)
        sample = sample.squeeze(1).cpu().numpy()

    return sample
