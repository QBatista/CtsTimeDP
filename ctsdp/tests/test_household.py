"""
Tests for `cts_household.py`.

"""

import numpy as np
from CtsTimeDP import Household
import scipy.io as spio
import os


TOL = 1e-3  # Matlab output is rounded to 4 digits

mat = spio.loadmat(os.path.join(os.path.dirname(__file__),
                                'data/consav_output.mat'))

γ = np.asscalar(mat['risk_aver'])
ρ = np.asscalar(mat['rho'])
r = np.asscalar(mat['r'])
μ_y = np.asscalar(mat['mu_y'])
σ_y = np.asscalar(mat['sd_y'])
num_y = np.asscalar(mat['ny'])
arrival_rate = np.asscalar(mat['arrivalrate_y'])
b_lim = np.asscalar(mat['borrow_lim'])
a_max = np.asscalar(mat['amax'])
num_a = np.asscalar(mat['na'])
α_grid = np.asscalar(mat['agrid_par'])

δ_hjb = np.asscalar(mat['delta_hjb'])
max_iter_hjb = np.asscalar(mat['maxiter_hjb'])
tol_hjb = np.asscalar(mat['tol_hjb'])

hh = Household(γ, ρ, r, μ_y, σ_y, num_y, arrival_rate, b_lim, a_max, num_a,
			   α_grid, verbose=False)

hh.solve_problem(δ_hjb, max_iter_hjb, tol_hjb)


def test_V():
    assert np.max(np.abs(hh.V - mat['V'].T)) < TOL


def test_g():
    assert np.max(np.abs(hh.g - mat['gmat'].T)) < TOL
