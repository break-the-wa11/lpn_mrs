import numpy as np
import pywt

def wv_denoise(x, wavelet='db10', threshold_factor=1.0):
    """
    Wavelet soft-thresholding denoising (global threshold).

    Parameters
    ----------
    x : np.ndarray, shape (b, n)
        Input noisy signals.
    wavelet : str
        Wavelet family.
    threshold_factor : float
        Multiplier on universal threshold: T = factor * sigma * sqrt(2 log N)

    Returns
    -------
    y : np.ndarray, shape (b, n)
        Denoised signals.
    """
    b, n = x.shape
    y_np = np.zeros_like(x)

    for i in range(b):
        sig = x[i]
        coeffs = pywt.wavedec(sig, wavelet)
        cA, details = coeffs[0], coeffs[1:]

        # Noise standard deviation from the finest detail subband
        sigma = np.median(np.abs(details[-1])) / 0.6745
        if sigma == 0:
            y_np[i] = sig
            continue

        # VisuShrink-style global threshold
        T = threshold_factor * sigma * np.sqrt(2 * np.log(n))

        # Threshold only detail coefficients
        details_th = [pywt.threshold(c, T, mode='soft') for c in details]
        coeffs_th = [cA] + details_th

        rec = pywt.waverec(coeffs_th, wavelet)
        y_np[i] = rec[:n]

    return y_np


def lpn_denoise(x, model):
    """
    Denoise using lpn

    Args:
    x: tensor, (b, *), * is the shape allowed by lpn model

    Returns:
    y: np.array, shape (b, n)
    """
    device = next(model.parameters()).device
    x = x.float().to(device)
    return model(x).squeeze(1).detach().cpu().numpy()

def lpn_cond_denoise(x, model, noise_levels):
    """
    Denoise using noise-conditional lpn

    Args:
    x: tensor, (b, *), * is the shape allowed by lpn model
    noise_levels: tensor, (b,1)

    Returns:
    y: np.array, shape (b, n)
    """
    device = next(model.parameters()).device
    x = x.float().to(device)
    noise_levels = noise_levels.float().to(device)
    return model(x, noise_levels).squeeze(1).detach().cpu().numpy()