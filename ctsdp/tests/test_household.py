"""
Tests for `cts_household.py`.

"""

import numpy as np
from ctsdp import Household
import scipy.io as spio
import os


TOL = 1e-3  # Matlab output is rounded to 4 digits

mat = spio.loadmat(os.path.join(os.path.dirname(__file__),
                                'data/consav_output.mat'))

γ = mat['risk_aver'].item()
ρ = mat['rho'].item()
r = mat['r'].item()
μ_y = mat['mu_y'].item()
σ_y = mat['sd_y'].item()
num_y = mat['ny'].item()
arrival_rate = mat['arrivalrate_y'].item()
b_lim = mat['borrow_lim'].item()
a_max = mat['amax'].item()
num_a = mat['na'].item()
α_grid = mat['agrid_par'].item()

δ_hjb = mat['delta_hjb'].item()
max_iter_hjb = mat['maxiter_hjb'].item()
tol_hjb = mat['tol_hjb'].item()

hh = Household(γ, ρ, r, μ_y, σ_y, num_y, arrival_rate, b_lim, a_max, num_a,
               α_grid, verbose=False)

hh.solve_problem(δ_hjb, max_iter_hjb, tol_hjb)


def test_V():
    assert np.max(np.abs(hh.V - mat['V'].T)) < TOL


def test_g():
    assert np.max(np.abs(hh.g_adj - mat['gmat'].T)) < TOL
