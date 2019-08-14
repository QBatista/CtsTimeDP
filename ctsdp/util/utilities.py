"""
A module containing utility functions for solving continuous time DP problems.

"""

import numpy as np
from scipy import stats


def discrete_normal(width, μ, σ, num):
    norm_cdf = stats.norm(μ, σ).cdf
    pts = np.linspace(μ - width * σ, μ + width * σ, num=num)
    pts = pts.reshape((-1, 1))

    probs = np.zeros(pts.shape)
    probs[0] = norm_cdf((pts[0] + pts[1]) / 2)
    for i in range(1, num - 1):
        probs[i] = (norm_cdf((pts[i+1] + pts[i]) / 2) -
                    norm_cdf((pts[i] + pts[i-1]) / 2))
    probs[-1] = 1 - probs[:-1].sum()

    μ_pts = probs.T @ pts
    σ_pts = np.sqrt(probs.T @ pts ** 2 - μ_pts ** 2)

    error = np.asscalar(σ_pts - σ)

    return np.abs(error), pts, probs


def CRRA_utility_function_factory(γ):
    if γ == 1:
        u = lambda c: np.log(c)
    else:
        u = lambda c: (c ** (1. - γ) - 1.) / (1. - γ)

    return u, lambda c: c ** -γ, lambda c: c ** (-1 / γ)


def verbose_decorator_factory(start_msg, end_msg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if args[0].verbose:  # Works for classes only
                print(start_msg)
                func(*args, **kwargs)
                print(end_msg)
            else:
                func(*args, **kwargs)
        return wrapper
    return decorator
