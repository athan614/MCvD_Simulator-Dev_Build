"""
Noise correlation analysis utilities for cross-channel correlation effects.

Provides vectorized calculations for residual DA-SERO correlation after
CTRL subtraction and charge-domain QNSI sensitivity analysis.
"""

import math
import numpy as np
from typing import Tuple, Union


def sigma_I_diff_vec(sigma_da: Union[float, np.ndarray], 
                     sigma_sero: Union[float, np.ndarray], 
                     rho: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Vectorized sigma_Q(diff) with residual DA-SERO correlation rho (scalar or ndarray).

    Args:
        sigma_da: DA channel noise standard deviation in charge domain (scalar or array)
        sigma_sero: SERO channel noise standard deviation in charge domain (scalar or array)
        rho: Cross-channel correlation coefficient (scalar or array)

    Returns:
        Combined differential noise standard deviation (scalar or array)
    """
    rho = np.clip(rho, -1.0, 1.0)
    var = sigma_da**2 + sigma_sero**2 - 2.0 * rho * sigma_da * sigma_sero
    return np.sqrt(np.maximum(var, 0.0))  # robust to tiny negatives from rounding


def onsi_curve(sigma_da: float, sigma_sero: float, 
               rho_min: float = -0.2, rho_max: float = 0.2, 
               n: int = 201) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns rho_grid, ONSI(ρ) = σ(I_diff; ρ)/σ(I_diff; 0).
    
    Args:
        sigma_da: DA channel noise (arbitrary units)
        sigma_sero: SERO channel noise (arbitrary units)
        rho_min: Minimum correlation to test
        rho_max: Maximum correlation to test
        n: Number of points in sweep
        
    Returns:
        Tuple of (rho_values, normalized_noise_values)
    """
    rhos = np.linspace(rho_min, rho_max, n)
    sigma0_raw = sigma_I_diff_vec(sigma_da, sigma_sero, 0.0)
    sigmas_raw = sigma_I_diff_vec(sigma_da, sigma_sero, rhos)
    
    # Ensure both are arrays for consistent return type
    sigma0 = np.asarray(sigma0_raw)
    sigmas = np.asarray(sigmas_raw)
    onsi = sigmas / sigma0
    
    return rhos, onsi


def onsi_curve_general(sigmas: np.ndarray, w: np.ndarray, rho_cc_grid: np.ndarray,
                      rho_gc: float = 0.0, rho_bc: float = 0.0) -> np.ndarray:
    """
    Generalized ONSI for tri-channel system with arbitrary linear combination weights.
    """
    sig = np.asarray(sigmas, dtype=float)
    w = np.asarray(w, dtype=float)
    
    def build_covariance_matrix(rho_gb: float) -> np.ndarray:
        """Build 3x3 covariance matrix with specified DA-SERO correlation."""
        S = np.diag(sig**2)
        S[0, 1] = S[1, 0] = rho_gb * sig[0] * sig[1]  # DA-SERO
        S[0, 2] = S[2, 0] = rho_gc * sig[0] * sig[2]  # DA-CTRL  
        S[1, 2] = S[2, 1] = rho_bc * sig[1] * sig[2]  # SERO-CTRL
        return S
    
    # Reference noise with no DA-SERO correlation (with edge-case protection)
    cov_matrix_0 = build_covariance_matrix(0.0)
    variance_0 = w @ cov_matrix_0 @ w
    eps = 1e-12  # FIX: Edge-case guard
    sigma0 = max(float(np.asarray(np.sqrt(np.maximum(variance_0, 0.0)))), eps)  # FIX: Robust conversion
    
    # Compute normalized noise for each correlation value
    result = []
    for rho_gb in rho_cc_grid:
        rho_gb_clipped = np.clip(rho_gb, -1.0, 1.0)
        cov_matrix = build_covariance_matrix(rho_gb_clipped)
        var = w @ cov_matrix @ w
        sigma = max(float(np.asarray(np.sqrt(np.maximum(var, 0.0)))), eps)  # FIX: Robust conversion
        result.append(sigma / sigma0)
    
    return np.asarray(result)

def optimal_ctrl_weight(cov_matrix: np.ndarray) -> float:
    """
    Calculate variance-minimizing CTRL weight β* for I_diff = I_DA - I_SERO - β*I_CTRL.
    
    Args:
        cov_matrix: 3x3 covariance matrix [DA, SERO, CTRL]
        
    Returns:
        Optimal β coefficient
    """
    # Extract relevant covariances
    cov_ctrl_da = cov_matrix[2, 0]  # Cov(I_CTRL, I_DA)
    cov_ctrl_sero = cov_matrix[2, 1]  # Cov(I_CTRL, I_SERO)
    var_ctrl = cov_matrix[2, 2]       # Var(I_CTRL)
    
    if var_ctrl < 1e-12:  # Avoid division by zero
        return 0.0
    
    # β* = Cov(I_CTRL, I_DA - I_SERO) / Var(I_CTRL)
    cov_ctrl_diff = cov_ctrl_da - cov_ctrl_sero
    beta_star = cov_ctrl_diff / var_ctrl
    
    return float(beta_star)

def onsi_curve_optimal_ctrl(sigmas: np.ndarray, rho_cc_grid: np.ndarray,
                           rho_gc: float = 0.0, rho_bc: float = 0.0) -> np.ndarray:
    """
    ONSI with optimal CTRL weighting: I_diff = I_DA - I_SERO - β*I_CTRL.
    """
    def build_cov(rho_gb: float) -> np.ndarray:
        sig = np.asarray(sigmas, dtype=float)
        S = np.diag(sig**2)
        S[0, 1] = S[1, 0] = rho_gb * sig[0] * sig[1]  # DA-SERO
        S[0, 2] = S[2, 0] = rho_gc * sig[0] * sig[2]  # DA-CTRL  
        S[1, 2] = S[2, 1] = rho_bc * sig[1] * sig[2]  # SERO-CTRL
        return S
    
    # Edge-case protection constant
    eps = 1e-12
    
    # Reference: optimal CTRL with no DA-SERO correlation
    cov_0 = build_cov(0.0)
    beta_0 = optimal_ctrl_weight(cov_0)
    w_0 = np.array([1.0, -1.0, -beta_0])
    variance_0 = w_0 @ cov_0 @ w_0
    sigma0 = np.sqrt(max(float(np.asarray(variance_0)), eps))  # FIX: Robust conversion
    
    # Compute ONSI for each correlation value
    result = []
    for rho_gb in rho_cc_grid:
        cov = build_cov(np.clip(rho_gb, -1.0, 1.0))
        beta = optimal_ctrl_weight(cov)
        w = np.array([1.0, -1.0, -beta])
        var = w @ cov @ w
        sigma = np.sqrt(max(float(np.asarray(var)), eps))  # FIX: Robust conversion
        result.append(sigma / sigma0)
    
    return np.asarray(result)


def compute_qnsi(
    delta_q_diff: Union[float, np.ndarray],
    sigma_da_Q: Union[float, np.ndarray],
    sigma_sero_Q: Union[float, np.ndarray],
    rho_post: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Compute charge-domain QNSI = DeltaQ_diff / sigma_Q_diff with clipping safeguards."""

    delta = np.asarray(delta_q_diff, dtype=float)
    sigma_da = np.asarray(sigma_da_Q, dtype=float)
    sigma_sero = np.asarray(sigma_sero_Q, dtype=float)
    rho = np.asarray(rho_post, dtype=float)

    sigma_q_diff = sigma_I_diff_vec(sigma_da, sigma_sero, rho)
    denom = np.maximum(sigma_q_diff, 1e-15)
    result = delta / denom

    if isinstance(result, np.ndarray):
        return np.asarray(result, dtype=float)
    if isinstance(result, (np.generic, float, int)):
        return float(result)
    raise TypeError(f"Unsupported QNSI result type: {type(result)!r}")
