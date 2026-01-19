"""
This file generate samples for multiple methods
"""

from sklearn.mixture import GaussianMixture

def GMM_sample(data, n_samples):
    """Fit GMM model to given data: np array"""
    gmm = GaussianMixture(n_components=1)
    gmm.fit(data)

    gmm_sample = gmm.sample(n_samples)[0]
    return gmm_sample
