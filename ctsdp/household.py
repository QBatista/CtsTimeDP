"""
A module for solving continuous-time consumption saving problems.

"""

import numpy as np
from scipy import sparse, optimize, interpolate
from ctsdp.util import (CRRA_utility_function_factory, discrete_normal,
                        verbose_decorator_factory)
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from IPython.core.display import display


start_hjb_fd_msg = \
    'Step 1: Solving the HJB equation using the finite difference method...'
end_hjb_msg = '...solved the HJB equation.'
start_kfe_linsolve_msg = \
    'Step 2: Solving the KFE by solving the linear system of equations...'
end_kfe_msg = '...solved the KFE.'
start_mpc_msg = 'Step 3: Computing marginal propensity to consume (MPC)...'
end_mpc_msg = '...computed MPCs.'

# TODO(QBatista):
# - Add documentation

class Household():
    """
    A class for representing households facing continuous-time DP consumption
    saving problems.

    """
    def __init__(self, γ=1., ρ=0.005275, r=0.005, μ_y=1., σ_y=0.25, num_y=5,
                 arrival_rate=0.25, b_lim=0., a_max=400., num_a=100,
                 α_grid=0.4, mpc_amount=1e-10, verbose=True):
        self.u, self.u1, self.u1_inv = CRRA_utility_function_factory(γ)
        self.γ = γ
        self.ρ = ρ
        self.r = r
        self.arrival_rate = arrival_rate
        self.num_y = num_y
        self.num_a = num_a
        self.n = num_y * num_a
        self.mpc_amount = mpc_amount

        self.hjb_solved = False
        self.kfe_solved = False

        self._create_income_grid(μ_y, σ_y, num_y)
        self._create_asset_grid(b_lim, a_max, num_a, α_grid)

        self.verbose = verbose

    def _create_income_grid(self, μ, σ, num):
        args = (μ, σ, num)

        x0 = 1.
        res = optimize.root(lambda x, *args: discrete_normal(x, *args)[0],
                            x0, args=args)

        if res.success:
            width = res.x
            _, self.y_grid, self.y_dist = discrete_normal(width, *args)
            self.y_grid_1d = self.y_grid.ravel()
            self.y_dist_1d = self.y_dist.ravel()
            y_trans = self.arrival_rate * (self.y_dist.T - np.eye(self.num_y))
            y_trans = np.kron(y_trans, np.eye(self.num_a))
            self.y_trans = sparse.csr_matrix(y_trans)

        else:
            raise ValueError('Failed to create grid for income.')

    def _create_asset_grid(self, b_lim, a_max, num, α):
        a_grid = np.linspace(0, 1, num=num) ** (1 / α)
        a_grid = b_lim + (a_max - b_lim) * a_grid
        self.a_grid_1d = a_grid
        self.a_grid = a_grid.reshape((1, -1))  # Chosen to match C-order

    def solve_problem(self, δ_hjb=1e10, max_iter_hjb=100, tol_hjb=1e-8,
                      δ_mpc=1e-2, cum_con_T=1.):
        self._solve_hjb_fd(δ_hjb, max_iter_hjb, tol_hjb)
        self._solve_kfe_linsolve()
        self._compute_mpc(δ_mpc, cum_con_T)

    @verbose_decorator_factory(start_hjb_fd_msg, end_hjb_msg)
    def _solve_hjb_fd(self, δ, max_iter, tol):
        # Asset grid for partial derivatives
        self.Δa_grid = np.diff(self.a_grid)
        Δa_grid_f = np.hstack([self.Δa_grid, self.Δa_grid[:, [-1]]])
        Δa_grid_b = np.hstack([self.Δa_grid[:, [0]], self.Δa_grid])

        con_0 = self.r * self.a_grid + self.y_grid
        H_0, dV_0 = self.u(con_0), self.u1(con_0)

        self.V = self.u(con_0) / self.ρ
        dV_f, dV_b = np.zeros_like(self.V), np.zeros_like(self.V)

        dV_f[:, -1] = self.u1(con_0[:, -1])
        dV_b[:, 0] = self.u1(con_0[:, 0])

        self.itr_hjb, self.error_hjb = 1, 1.

        while not self.hjb_solved and self.itr_hjb <= max_iter:
            V_diff = np.diff(self.V)

            # Forward difference
            dV_f[:, :-1] = V_diff / Δa_grid_f[:, :-1]

            # Backward difference
            dV_b[:, 1:] = V_diff / Δa_grid_b[:, 1:]

            # Consumption and savings with forward difference
            con_f = self.u1_inv(dV_f)
            sav_f = con_0 - con_f
            H_f = self.u(con_f) + dV_f * sav_f

            # Consumption and savings with backward difference
            con_b = self.u1_inv(dV_b)
            sav_b = con_0 - con_b
            H_b = self.u(con_b) + dV_b * sav_b

            I_neither = (1 - (sav_f > 0)) * (1 - (sav_b < 0))
            I_unique = ((sav_b < 0) * (1 - (sav_f > 0)) +
                        (1 - (sav_b < 0)) * (sav_f > 0))
            I_both = (sav_b < 0) * (sav_f > 0)
            I_b = (I_unique * (sav_b < 0) * (H_b > H_0) +
                   I_both * (H_b > H_f) * (H_b > H_0))
            I_f = (I_unique * (sav_f > 0) * (H_f > H_0) +
                   I_both * (H_f > H_b) * (H_f > H_0))
            I_0 = 1 - I_b - I_f

            self.con = con_f * I_f + con_b * I_b + con_0 * I_0
            self.sav = sav_f * I_f + sav_b * I_b
            util = self.u(self.con)

            A_diag_b = I_b * sav_b / Δa_grid_b
            A_diag_f = I_f * sav_f / Δa_grid_f

            A_low_diag = -A_diag_b
            A_diag = -A_diag_f + A_diag_b
            A_up_diag = A_diag_f

            A_low_diag[:, 0] = 0.
            A_up_diag[:, -1] = 0.

            self.A = (sparse.diags(A_diag.ravel()) +
                      sparse.diags(A_low_diag.ravel()[1:], offsets=-1) +
                      sparse.diags(A_up_diag.ravel()[:-1], offsets=1) +
                      self.y_trans)

            A_hjb = (self.ρ + 1 / δ) * sparse.eye(self.n) - self.A
            b_hjb = util.ravel() + self.V.ravel() / δ

            V_new = sparse.linalg.spsolve(A_hjb, b_hjb).reshape(self.V.shape)

            self._update_hjb(V_new, tol)

        if not self.hjb_solved:
            raise ValueError('Failed to solve HJB equation.')

    def _update_hjb(self, V_new, tol):
        self.error_hjb = np.max(np.abs(V_new - self.V))
        self.hjb_solved = self.error_hjb < tol
        self.V = V_new

        if self.verbose and (self.itr_hjb % 5 == 0 or self.hjb_solved):
            print('Iteration', self.itr_hjb, ': max HJB error is',
                  self.error_hjb)

        self.itr_hjb += 1

    @verbose_decorator_factory(start_kfe_linsolve_msg, end_kfe_msg)
    def _solve_kfe_linsolve(self, rel_sol_constant=1.):
        # Trapezoidal rule for KFE and moments
        a_δ = np.zeros((1, self.num_a))
        a_δ[:, 0] = 0.5 * self.Δa_grid[:, 0]
        a_δ[:, 1:-1] = 0.5 * self.Δa_grid[:, :-1] + 0.5 * self.Δa_grid[:, 1:]
        a_δ[:, -1] = 0.5 * self.Δa_grid[:, -1]
        self.a_δ = a_δ

        A_g = self.A.T
        b_g = np.zeros(self.n)
        b_g[0] = rel_sol_constant

        g = sparse.linalg.spsolve(A_g, b_g)
        g = g / g.sum()
        self.g = g.reshape((self.num_y, self.num_a))
        self.g_adj = self.g / a_δ

        a_cdf = (self.g_adj.sum(axis=0) * self.a_δ).cumsum()
        self.interp_cdf = interpolate.PchipInterpolator(x=self.a_grid_1d,
                                                        y=a_cdf)

        self.kfe_solved = True

    @verbose_decorator_factory(start_mpc_msg, end_mpc_msg)
    def _compute_mpc(self, δ_mpc, cum_con_T):
        cum_con = np.zeros(self.n)

        A_mpc = (sparse.eye(self.n) / δ_mpc - self.A)
        for i in np.arange(1., 0. - δ_mpc, -δ_mpc):
            b_mpc = self.con.ravel() + cum_con / δ_mpc
            cum_con = sparse.linalg.spsolve(A_mpc, b_mpc)

        cum_con = cum_con.reshape(self.V.shape)
        self.mpc = np.zeros_like(self.V)

        for y_i in range(self.num_y):
            interp_f = interpolate.PchipInterpolator(x=self.a_grid_1d,
                                                     y=cum_con[y_i],
                                                     extrapolate=True)

            self.mpc[y_i] = (interp_f(self.a_grid + self.mpc_amount) -
                             cum_con[y_i]) / self.mpc_amount

        self.mpc_lim = cum_con_T * ((self.ρ - self.r) / self.γ + self.r)

    def plot_problem_solution(self, height=1300, width=900):
        colors = ['#1f77b4',  # muted blue
                  '#ff7f0e',  # safety orange
                  '#2ca02c',  # cooked asparagus green
                  '#d62728',  # brick red
                  '#9467bd',  # muted purple
                  '#8c564b',  # chestnut brown
                  '#e377c2',  # raspberry yogurt pink
                  '#7f7f7f',  # middle gray
                  '#bcbd22',  # curry yellow-green
                  '#17becf'   # blue-teal
                  ]

        subplot_titles = ['Value Function',
                          'Income Distribution',
                          'Optimal Consumption Policy Function',
                          'Optimal Savings Policy Function',
                          'Stationary Asset Distribution',
                          'Marginal Propensity to Consume (MPC)',
                          'Distribution of MPC']

        subplots = make_subplots(rows=4, cols=2, subplot_titles=subplot_titles,
                                 specs=[[{}, {}],
                                        [{}, {}],
                                        [{"colspan": 2}, None],
                                        [{}, {}]])

        fig = go.FigureWidget(subplots)

        # Stationary Asset Distribution
        bin_size = 0.01
        bins = np.arange(self.a_grid_1d[0], self.a_grid_1d[-1], bin_size)

        a_hist = np.zeros(len(bins))
        a_hist[0] = self.interp_cdf(bins[0])
        a_hist[1:] = self.interp_cdf(bins[1:]) - self.interp_cdf(bins[:-1])

        fig.add_trace(go.Scatter(x=bins,
                                 y=a_hist,
                                 name='Asset Dist.',
                                 showlegend=False),
                      row=3, col=1)

        # Income Distribution
        fig.add_trace(go.Bar(x=self.y_grid_1d,
                             y=self.y_dist_1d,
                             showlegend=False,
                             name='Inc. Dist.'),
                      row=1, col=2)

        # MPC Distribution
        bin_size = 0.02
        bins = np.arange(0., 1., bin_size)
        hist, bins = np.histogram(self.mpc, bins=bins, weights=self.g)

        fig.add_trace(go.Bar(x=bins,
                             y=hist,
                             showlegend=False,
                             name='MPC Dist.'),
                      row=4, col=2)

        objects = [(1, 1, False, self.V),  # Optimal Value Function
                   (2, 1, True, self.con),  # Optimal Consumption Policy
                   (2, 2, False, self.sav),  # Optimal Savings Policy
                   (4, 1, False, self.mpc)]  # MPC Amount

        for i in range(self.num_y):
            y_i = ('y_' + str(i+1) + ' = ' +
                   str(np.asscalar(self.y_grid[i].round(4))))

            for row_i, col_i, showlegend, obj in objects:
                fig.add_trace(go.Scatter(x=self.a_grid_1d,
                                         y=obj[i],
                                         legendgroup=y_i,
                                         name=y_i,
                                         showlegend=showlegend,
                                         line={'color': colors[i]}),
                              row=row_i, col=col_i)


        # Theoretical MPC
        mpc_lim_data = np.array([self.mpc_lim] * self.num_a)
        fig.add_trace(go.Scatter(x=self.a_grid_1d, y=mpc_lim_data,
                                 name='Theoretical MPC',
                                 line={'color': 'black', 'dash':'dash'}),
                      row=4, col=1)

        fig.update_layout(height=height, width=width,
                          title_text='Household Problem Solution Plots')

        display(fig)
