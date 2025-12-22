"""
Laplace Mechanism Implementation for Differential Privacy.

This module provides secure implementations of the Laplace mechanism,
including discrete sampling to mitigate floating-point vulnerabilities.
"""

import numpy as np
from typing import Optional
import secrets


def generate_laplace_noise(scale: float, size: Optional[int] = None) -> np.ndarray | float:
    """
    Generate Laplace noise with the given scale parameter.
    
    The Laplace distribution has PDF: f(x) = (1/2λ) * exp(-|x|/λ)
    where λ (lambda/scale) determines the spread.
    
    Args:
        scale: The scale parameter λ of the Laplace distribution.
               For ε-differential privacy, scale = sensitivity/ε
        size: Optional number of samples to generate
        
    Returns:
        A single float or array of Laplace-distributed noise values.
    """
    if scale <= 0:
        raise ValueError("Scale parameter must be positive")
    
    # Use numpy's built-in Laplace for efficiency
    return np.random.laplace(loc=0.0, scale=scale, size=size)


def discrete_laplace_sample(scale: float, granularity: float = 1e-10) -> float:
    """
    Generate discrete Laplace noise to prevent floating-point attacks.
    
    Standard floating-point Laplace sampling can leak information through
    the precision of the returned values. This implementation discretizes
    the output to a fixed granularity.
    
    Args:
        scale: The scale parameter λ of the Laplace distribution
        granularity: The discretization step size
        
    Returns:
        A discretized Laplace noise value
    """
    # Generate standard Laplace noise
    noise = generate_laplace_noise(scale)
    
    # Discretize to fixed granularity
    return round(noise / granularity) * granularity


def secure_laplace_sample(scale: float) -> float:
    """
    Generate Laplace noise using cryptographically secure randomness.
    
    Uses the inverse CDF method with secure random bytes to ensure
    unpredictability even against adversaries with side-channel access.
    
    Args:
        scale: The scale parameter λ of the Laplace distribution
        
    Returns:
        A securely sampled Laplace noise value
    """
    # Generate 64 bits of secure randomness
    random_bytes = secrets.token_bytes(8)
    # Convert to uniform [0, 1) value
    u = int.from_bytes(random_bytes, 'big') / (2**64)
    
    # Shift to (-0.5, 0.5) to handle both tails
    u = u - 0.5
    
    # Inverse CDF of Laplace: F^(-1)(p) = -λ * sign(p - 0.5) * ln(1 - 2|p - 0.5|)
    if u >= 0:
        return -scale * np.log(1 - 2 * u)
    else:
        return scale * np.log(1 + 2 * u)


def calculate_noise_scale(sensitivity: float, epsilon: float) -> float:
    """
    Calculate the Laplace noise scale for ε-differential privacy.
    
    For the Laplace mechanism to satisfy ε-DP, we need:
        scale = sensitivity / epsilon
    
    Args:
        sensitivity: The L1 sensitivity of the query
        epsilon: The privacy budget (smaller = more private)
        
    Returns:
        The scale parameter for Laplace noise generation
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    if sensitivity < 0:
        raise ValueError("Sensitivity must be non-negative")
        
    return sensitivity / epsilon


def add_laplace_noise(value: float, sensitivity: float, epsilon: float) -> float:
    """
    Add Laplace noise to a value to achieve ε-differential privacy.
    
    Args:
        value: The true value to protect
        sensitivity: The L1 sensitivity of the query
        epsilon: The privacy budget
        
    Returns:
        The noisy value
    """
    scale = calculate_noise_scale(sensitivity, epsilon)
    noise = generate_laplace_noise(scale)
    return value + noise

