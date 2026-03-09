from typing import Tuple
import numpy as np
from skimage.restoration import denoise_tv_chambolle
import cv2
from ..session import OIDIQSession, OIDIQPreprocessor, PreProcessors
from ..utils import OIDIQConfig, creates, config, QualityMetricConfig


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise soft-thresholding."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def tv_proximal(v: np.ndarray, weight: float) -> np.ndarray:
    """
    Proximal operator for TV: prox_{weight * TV}(v)
    Implemented by calling skimage.restoration.denoise_tv_chambolle which solves:

        argmin_u 0.5 * ||u - v||_2^2 + weight * TV(u)

    weight > 0 is the regularization parameter.
    """
    return denoise_tv_chambolle(v, weight=weight, channel_axis=-1)


def admm_sparse_tv(
    session: OIDIQSession,
    I: np.ndarray,
    lambda_tv: float = 0.5,
    lambda_s: float = 0.05,
    rho: float = 1.0,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:


    S = np.zeros_like(I)  # sparse component
    G = np.copy(I)  # auxiliary variable for B's TV prox
    Y = np.zeros_like(I)  # scaled dual (we'll use scaled form: Y = U)
    B = np.copy(I)  # background estimate

    for k in range(1, max_iter + 1):
        # B-update (quadratic closed form)
        numerator = I - S + rho * (G - Y)
        B = numerator / (1.0 + rho)

        # G-update: proximal for TV
        V = B + Y  # argument to prox
        # weight passed to denoise_tv_chambolle should be lambda_tv / rho
        weight = lambda_tv / rho
        
        G = tv_proximal(V, weight=weight)
        # S-update: soft-thresholding of (I - B)
        Z = I - B
        S = soft_threshold(Z, lambda_s)

        # Dual update (scaled dual variable)
        Y = Y + B - G

        # diagnostics: primal residual r = B - G
        r_norm = np.linalg.norm((B - G).ravel())

        if r_norm < tol:
            session.log(ForegroundMasking, f"ADMM converged in {k} iterations with residual {r_norm:.6f}.")
            break

    return B, S


def foreground_mask_from_S(S: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Compute a binary foreground mask from sparse component S.

    Returns a boolean mask of foreground pixels.
    """
    absS = np.abs(S)
    if S.ndim == 3:
        # combine channels by max
        absS_comb = np.max(absS, axis=2)
    else:
        absS_comb = absS

    med = np.median(absS_comb)
    std = np.std(absS_comb)
    threshold = med + 2.0 * std
    mask = absS_comb > threshold

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)

    return mask


class ForegroundMasking(OIDIQPreprocessor):
    @creates(PreProcessors.NORMALIZED_FOREGROUND_MASK)
    @config("foreground_masking")
    def process(self, session: OIDIQSession, config: OIDIQConfig) -> np.ndarray:
        img = session.get_normalized_image()
        div_size = config.get("downsample_divisor", 1)
        if div_size > 1:
            img = cv2.resize(img, (img.shape[1] // div_size, img.shape[0] // div_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        B, S = admm_sparse_tv(
            I=img,
            lambda_tv=config.get("lambda_tv", 0.5),
            lambda_s=config.get("lambda_s", 0.05),
            max_iter=config.get("max_iters", 5),
            rho=config.get("rho", 1.0),
            tol=config.get("tol", 1e-4),
            session=session,
        )
        if div_size > 1:
            S = cv2.resize(S, (img.shape[1] * div_size, img.shape[0] * div_size), interpolation=cv2.INTER_CUBIC)
        return foreground_mask_from_S(S, kernel_size=config.get("kernel_size", 3), iterations=config.get("iterations", 1))

