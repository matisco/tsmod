import numpy as np
# from mpmath.libmp.libmpf import negative_rnd
# from numba.core.ir import Raise
from functools import cached_property  # wraps
from typing import Literal, Optional  # Iterable

# from numpy.linalg import matrix_rank  # use to check stationarity
from scipy.linalg import eigh, qr, cho_factor, cho_solve  # used for exact diffuse, kalman filter uses inv (questionable choice)
from numba import njit

from optimization_objectives import OptimizationObjective, GaussianNLL
from state_space.linear.representation import MutableLinearStateSpaceModelRepresentation, LinearStateSpaceModelRepresentation

# from dynamax.linear_gaussian_ssm.inference import (lgssm_filter,
#                                                    PosteriorGSSMFiltered,
#                                                    ParamsLGSSM,
#                                                    ParamsLGSSMInitial,
#                                                    ParamsLGSSMDynamics,
#                                                    ParamsLGSSMEmissions,
#                                                    )
# from jaxtyping import Float
# import jax
# import jax.numpy as jnp


# TODO: 1. Square root filtering i want
#       2. I clearly didnt know how to code when i did this. Forgive me. It works


@njit  # this should be the main work horse here
def kalman_nll_SS(demeaned_endog, F, Z, H, kalman_gain_SS, innovation_precision, P_SS, x0):

    T = demeaned_endog.shape[0]

    x_pred = F @ x0

    nll = 0.0

    pred_error = demeaned_endog[0] - Z @ x_pred
    nll += pred_error @ (innovation_precision @ pred_error)
    for t in range(1, T):
        x_pred = F @ (x_pred + kalman_gain_SS @ pred_error)
        pred_error = demeaned_endog[t] - Z @ x_pred
        nll += pred_error @ (innovation_precision @ pred_error)

    nll = nll/ 2

    S = Z @ P_SS @ Z.T + H
    _, logdet = np.linalg.slogdet(S)
    const_term = 0.5 * np.log(2 * np.pi) * Z.shape[0]
    nll += (const_term + logdet/2) * T

    return nll


@njit
def kalman_pred_error_SS(demeaned_endog, F, Z, kalman_gain_SS, x0):
    T = demeaned_endog.shape[0]

    prediction_errors = np.zeros_like(demeaned_endog)

    x_pred = F @ x0
    prediction_errors[0] = demeaned_endog[0] - Z @ x_pred
    for t in range(1, T):
        x_pred = F @ (x_pred + kalman_gain_SS @ prediction_errors[t - 1])
        prediction_errors[t] = demeaned_endog[t] - Z @ x_pred

    return prediction_errors


@njit
def kalman_filter_SS(y, mu, Z, H, F, x0, steady_state_covariance, steady_state_kalman_gain, steady_state_obs_precision):
    T = y.shape[0]
    d = x0.shape[0]  # Dimension of state
    if y.ndim > 1:
        N = y.shape[1]  # Dimension of observations
    else:
        N = 1

    demeaned_endog = y - mu

    # Initialize arrays
    x_predicted = np.zeros((T, d))
    pred_error = np.zeros((T, N))

    nll = 0.0

    x_predicted[0] = F @ x0
    pred_error[0] = demeaned_endog[0] - Z @ x_predicted[0]
    nll += pred_error[0].T @ steady_state_obs_precision @ pred_error[0]
    for t in range(1, T):
        x_predicted[t] = F @ (x_predicted[t - 1] + steady_state_kalman_gain @ pred_error[t - 1])
        pred_error[t] = demeaned_endog[t] - Z @ x_predicted[t]
        nll += pred_error[t].T @ steady_state_obs_precision @ pred_error[t]

    nll = nll / 2
    nll_constant = np.log(2 * np.pi) * N
    S = Z @ steady_state_covariance @ Z.T + H
    _, logdet = np.linalg.slogdet(S)
    nll += (logdet + nll_constant) * T / 2

    return nll, pred_error, x_predicted


@njit
def kalman_filter(y, mu, Z, H, F, Q, x0, P0):
    """
    Kalman filter for a linear Gaussian state-space model.

    """
    T = y.shape[0]
    d = x0.shape[0]  # Dimension of state
    if y.ndim > 1:
        N = y.shape[1]  # Dimension of observations
    else:
        N = 1

    I_d = np.eye(d)
    demeaned_endog = y - mu

    # Initialize arrays
    pred_error = np.zeros((T, N))
    x_predicted = np.zeros((T, d))
    P_predicted = np.zeros((T, d, d))
    ZTinvS = np.zeros((T, d, N))

    nll = 0.0

    x_predicted[0] = F @ x0
    P_predicted[0] = F @ P0 @ F.T + Q
    for t in range(T):
        if t > 0:
            kalman_gain = P_predicted[t - 1] @ ZTinvS[t - 1]
            x_predicted[t] = F @ (x_predicted[t - 1] + kalman_gain @ pred_error[t - 1])
            P_predicted[t] = F @ (I_d - kalman_gain @ Z) @ P_predicted[t - 1] @ F.T + Q
            P_predicted[t] = (P_predicted[t] + P_predicted[t].T) / 2

        pred_error[t] = demeaned_endog[t] - Z @ x_predicted[t]
        S = Z @ P_predicted[t] @ Z.T + H

        # option 1
        pred_error_precision = np.linalg.inv(S)
        logdet = np.log(np.linalg.det(S))
        quad_form = pred_error[t].T @ pred_error_precision @ pred_error[t]
        ZTinvS[t] = Z.T @ pred_error_precision

        # option 2
        # L = np.linalg.cholesky(S)  # S = L L^T
        # logdet = 2.0 * np.sum(np.log(np.diag(L)))
        # y = np.linalg.solve(L, pred_error[t])
        # quad_form = y.T @ y
        # ZTinvS[t] = np.linalg.solve(L.T, np.linalg.solve(L, H)).T  # H^T S^-1

        nll += logdet + quad_form

    nll = nll / 2
    nll_constant = np.log(2 * np.pi) * N
    nll += nll_constant * T / 2

    return nll, pred_error, x_predicted, P_predicted, ZTinvS


@njit
def kalman_filter_exact_diffuse_univariate(y, mu, Z, H, F, Q, x0, P_star, P_infty):
    """
    Kalman filter for a linear Gaussian state-space model.

    Source: Siam Jan Koopman, 1997, Exact Initial Kalman Filtering and Smoothing for Nonstationary Time Series Models

    """
    T_steps = y.shape[0]
    d = x0.shape[0]  # Dimension of state
    N = 1  # Dimension of observations = 1

    # Initialize arrays
    x_predicted = np.zeros((T_steps, d))
    P_predicted_star = np.zeros((T_steps, d, d))
    P_predicted_infty = np.zeros((T_steps, d, d))

    innovation = np.zeros((T_steps, N))

    ZTSinv_star = np.zeros((T_steps, d, N))
    ZTSinv_infty = np.zeros((T_steps, d, N))

    nll = np.float64(0.0)

    x_predicted[0] = F @ x0
    P_predicted_star[0] = F @ P_star @ F.T + Q
    P_predicted_infty[0] = F @ P_infty @ F.T

    default_kalman = False
    t_diffuse = 0
    demeaned_endog = y - mu

    nll_constant = np.float64(np.log(2 * np.pi) * N)

    C_star = np.zeros((d, d))
    C_infty = np.zeros((d, d))
    kalman_gain = np.zeros((d, N))
    F_star_minus = np.float64(0.0)
    F_infty_minus = np.float64(0.0)

    # Forward pass: Kalman Filter
    for t in range(T_steps):
        if t > 0:
            x_predicted[t] = F @ (x_predicted[t - 1] + kalman_gain @ innovation[t - 1])

            P_predicted_star[t] = F @ (P_predicted_star[t - 1] - C_star) @ F.T + Q
            P_predicted_star[t] = (P_predicted_star[t] + P_predicted_star[t].T) / 2

            P_predicted_infty[t] = F @ (P_predicted_infty[t - 1] - C_infty) @ F.T
            P_predicted_infty[t] = (P_predicted_infty[t] + P_predicted_infty[t].T) / 2

        F_star = (Z @ P_predicted_star[t] @ Z.T + H)[0,0]
        F_infty = (Z @ P_predicted_infty[t] @ Z.T)[0,0]

        M_star = P_predicted_star[t] @ Z.T
        M_infty = P_predicted_infty[t] @ Z.T

        if not default_kalman:
            if F_infty < 1e-6:
                default_kalman = True
                t_diffuse = t

        if default_kalman:
            kalman_gain = M_star / F_star
            C_star = M_star @ kalman_gain.T
            C_infty = np.zeros((d, d))
        else:
            kalman_gain = M_infty / F_infty
            C_star = M_star @ kalman_gain.T + kalman_gain @ (M_star - kalman_gain * F_star).T
            C_infty = M_infty @ M_infty.T / F_infty

        innovation[t] = demeaned_endog[t] - Z @ x_predicted[t]
        # x_updated[t] = x_predicted[t] + P_predicted_star[t] @ Z.T @ F_star_minus[t] @ innovation[t]

        if default_kalman:
            F_star_minus = 1 / F_star
            ZTSinv_star[t] = Z.T * F_star_minus
        else:
            F_infty_minus = 1 / F_infty
            ZTSinv_infty[t] = Z.T * F_infty_minus

        logdet = np.log((F_star_minus + F_infty_minus))
        nll += (nll_constant - logdet + innovation[t, 0] * F_star_minus * innovation[t, 0])

    nll = nll / 2
    return nll, innovation, x_predicted, P_predicted_star, P_predicted_infty, ZTSinv_star, ZTSinv_infty, t_diffuse


def kalman_filter_exact_diffuse(y, mu, Z, H, F, Q, x0, P_star, P_infty):
    """
    Kalman filter for a linear Gaussian state-space model.

    Source: Siam Jan Koopman, 1997, Exact Initial Kalman Filtering and Smoothing for Nonstationary Time Series Models
    """
    T_steps = y.shape[0]
    d = x0.shape[0]  # Dimension of state
    if y.ndim > 1:
        N = y.shape[1]  # Dimension of observations
    else:
        N = 1

    if N == 1:  # avoids simultaneous congruence without hassle
        nll, pred_error, x_predicted, P_predicted_star, P_predicted_infty, ZTSinv_star, ZTSinv_infty, t_diffuse =\
            kalman_filter_exact_diffuse_univariate(y, mu, Z, H, F, Q, x0, P_star, P_infty)

        return (nll, pred_error, x_predicted,
                PowerExpansionArray(P_predicted_star, P_predicted_infty, t_diffuse),
                PowerExpansionArray(ZTSinv_star, ZTSinv_infty, t_diffuse))

    # Initialize arrays
    pred_error = np.zeros((T_steps, N))
    x_predicted = np.zeros((T_steps, d))
    P_predicted_star = np.zeros((T_steps, d, d))
    ZTSinv_star = np.zeros((T_steps, d, N))

    P_predicted_infty = []
    ZTSinv_infty = []

    nll = 0.0

    x_predicted[0] = F @ x0
    P_predicted_star[0] = F @ P_star @ F.T + Q
    P_predicted_infty.append(F @ P_infty @ F.T)

    t_diffuse = None

    C_star = np.zeros((d, d))
    C_infty = np.zeros((d, d))

    # x_updated[t] = x_predicted[t] + P_predicted[t] @ ZTinvS[t] @ prediction_error[t]
    # P_updated[t] = (I_d - kalman_gain @ Z) @ P_predicted[t]
    nll_constant = np.log(2 * np.pi) * N

    def _calc_diffuse_over_j1_j2(F_star, F_infty):
        diffuse_is_over = False
        if np.linalg.matrix_rank(F_infty, tol=1e-10) == 0:
            diffuse_is_over = True

        if not diffuse_is_over:
            J1, J2 = simultaneous_congruence(F_infty, F_star, tol=1e-10)
            if J2 is None:
                diffuse_is_over = True
        else:
            J1, J2 = None, None

        return diffuse_is_over, J1, J2

    kalman_gain = np.zeros((d, N))
    for t in range(T_steps):
        if t > 0:
            x_predicted[t] = F @ (x_predicted[t - 1] + kalman_gain @ pred_error[t - 1])

            P_predicted_star[t] = F @ (P_predicted_star[t - 1] - C_star) @ F.T + Q
            P_predicted_star[t] = (P_predicted_star[t] + P_predicted_star[t].T) / 2

            P_predicted_infty.append(F @ (P_predicted_infty[-1] - C_infty) @ F.T)

        pred_error[t] = y[t] - mu - Z @ x_predicted[t]

        F_star = Z @ P_predicted_star[t] @ Z.T + H
        F_infty = Z @ P_predicted_infty[-1] @ Z.T

        diffuse_is_over, J1, J2 = _calc_diffuse_over_j1_j2(F_star, F_infty)

        if diffuse_is_over:
            # update and call normal kalman filter

            F_star_minus = np.linalg.inv(F_star)
            ZTSinv_star[t] = Z.T @ np.linalg.inv(F_star)

            M_star = P_predicted_star[t] @ Z.T
            kalman_gain = M_star @ F_star_minus
            C_star = M_star @ kalman_gain.T

            logdet = np.log(np.linalg.det(F_star_minus))
            nll += 0.5 * (nll_constant - logdet + pred_error[t].T @ F_star_minus @ pred_error[t])

            t_diffuse = t

            x_updated = x_predicted[t] + kalman_gain @ pred_error[t]
            P_updated = P_predicted_star[t] - C_star

            if t_diffuse < T_steps - 1:
                nll_, pred_error_, x_predicted_, P_predicted_, ZTSinv_ =\
                    kalman_filter(y[t_diffuse + 1:T_steps], mu, Z, H, F, Q, x_updated, P_updated)

                nll += nll_
                pred_error[t_diffuse + 1:T_steps] = pred_error_
                x_predicted[t_diffuse + 1:T_steps] = x_predicted_
                P_predicted_star[t_diffuse + 1:T_steps] = P_predicted_
                ZTSinv_star[t_diffuse + 1:T_steps] = ZTSinv_

            break

        else:
            F_star_minus = J2 @ J2.T
            F_infty_minus = J1 @ J1.T

            ZTSinv_star[t] = Z.T @ F_star_minus
            ZTSinv_infty.append(Z.T @ F_infty_minus)

            M_star = P_predicted_star[t] @ Z.T
            M_infty = P_predicted_infty[-1] @ Z.T

            kalman_gain = M_star @ F_star_minus + M_infty @ F_infty_minus

            C_star = M_star @ kalman_gain.T + M_infty @ F_infty_minus @ (
                    M_star - M_infty @ F_infty_minus @ F_star).T

            C_infty = M_infty @ F_infty_minus @ M_infty.T

            logdet = np.log(np.linalg.det(F_star_minus + F_infty_minus))
            nll += 0.5 * (nll_constant - logdet + pred_error[t].T @ F_star_minus @ pred_error[t])


    P_predicted_infty = np.stack(P_predicted_infty, axis=0)
    ZTSinv_infty = np.stack(ZTSinv_infty, axis=0)

    return (nll, pred_error, x_predicted,
            PowerExpansionArray(P_predicted_star, P_predicted_infty, t_diffuse),
            PowerExpansionArray(ZTSinv_star, ZTSinv_infty, t_diffuse))


@njit
def kalman_smoother(Z, F, pred_error, x_predicted, P_predicted, ZTinvS, t0=0):
    """
    Kalman smoother for a linear Gaussian state-space model.

    """
    is_steady_state = P_predicted.ndim == 2

    T = x_predicted.shape[0]
    d = F.shape[0]  # Dimension of state

    I_d = np.eye(d)

    # Backward pass: RTS Smoother
    x_smooth = np.zeros((T - t0, d))
    P_smooth = np.zeros((T - t0, d, d))
    Plag_smooth = np.zeros((T - t0, d, d))  # Lag-one covariance

    r_t = np.zeros(d)
    N_t = np.zeros((d, d))

    if is_steady_state:
        P_predicted_t_prev = P_predicted
        P_predicted_t = P_predicted
        ZTinvS_t = ZTinvS
    else:
        P_predicted_t = P_predicted[T - 1]
        ZTinvS_t = ZTinvS[T - 1]

    kalman_gain = P_predicted_t @ ZTinvS_t
    L_t = F @ (I_d - kalman_gain @ Z)
    # for t in reversed(range(t0, T)):
    for t in range(T - 1, t0 - 1, -1):
        r_prev = ZTinvS_t @ pred_error[t] + L_t.T @ r_t
        N_prev = ZTinvS_t @ Z + L_t.T @ N_t @ L_t

        x_smooth[t - t0] = x_predicted[t] + P_predicted_t @ r_prev
        P_smooth[t - t0] = P_predicted_t - P_predicted_t @ N_prev @ P_predicted_t
        P_smooth[t - t0] = (P_smooth[t - t0] + P_smooth[t - t0].T) / 2

        if not is_steady_state:
            ZTinvS_t = ZTinvS[t - 1]
            P_predicted_t_prev = P_predicted[t]
            P_predicted_t = P_predicted[t - 1]

        kalman_gain = P_predicted_t_prev @ ZTinvS_t
        L_t = F @ (I_d - kalman_gain @ Z)
        r_t = r_prev
        N_t = N_prev
        Plag_smooth[t - t0] = (I_d - N_prev @ P_predicted_t_prev).T @ L_t @ P_predicted_t

    if t0 == 0:
        Plag_smooth[t0] = np.zeros((d, d))  # No lag at t=0

    return x_smooth, P_smooth, Plag_smooth



def _kalman_updater(predicted: "KalmanFilterResult"):  # TODO: not good atm. i dont think it will be used much

    prediction_error = predicted.prediction_errors
    x_predicted = predicted.predicted_state
    P_predicted = predicted.predicted_state_cov

    ZTinvS = predicted._ZTinvS

    T = x_predicted.shape[0]
    d = predicted.representation.F.shape[0]  # Dimension of state

    # Initialize arrays
    x_updated = np.zeros_like(x_predicted)

    I_d = np.eye(d)

    Z = predicted.representation.Z
    for t in range(T):
        x_updated[t] = x_predicted[t] + P_predicted[t] @ ZTinvS[t] @ prediction_error[t]

    m = [P_predicted, ZTinvS]
    if all(isinstance(mat, SteadyStateArray) for mat in m):
        kalman_gain = P_predicted[0] @ ZTinvS[0]
        P_updated = (I_d - kalman_gain @ Z) @ P_predicted[0]
        P_updated = SteadyStateArray(P_updated, T)
    elif any(isinstance(mat, SteadyStateArray) for mat in m):
        raise RuntimeError("Whuuuuuuuut?")
    else:
        P_updated = np.zeros_like(P_predicted)
        for t in range(T):
            kalman_gain = P_predicted[t - 1] @ ZTinvS[t - 1]
            P_updated[t] = (I_d - kalman_gain @ Z) @ P_predicted[t]
            P_updated[t] = (P_updated[t] + P_updated[t].T) / 2

    return x_updated, P_updated

# def _is_stationary(A, C, Q, R, tol=1e-8):
#     """
#     Checks if a discrete-time system is stationary for steady-state Kalman filtering.
#
#     Parameters
#     ----------
#     A : np.ndarray
#         State transition matrix (n x n)
#     C : np.ndarray
#         Observation matrix (m x n)
#     Q : np.ndarray
#         Process noise covariance (n x n)
#     R : np.ndarray
#         Measurement noise covariance (m x m)
#     tol : float
#         Tolerance for rank checks
#
#     Returns
#     -------
#     stationary : bool
#         True if the system satisfies stationarity conditions
#     info : dict
#         Detailed diagnostic information
#     """
#     n = A.shape[0]
#     stationary = True
#     info = {'unstable_unobservable': [], 'unstable_observable': [], 'stabilizable': True}
#
#     # Eigenvalues of A
#     eigvals = np.linalg.eigvals(A)
#
#     # Cholesky-like square root of Q for controllability check
#     try:
#         Q_sqrt = np.linalg.cholesky(Q)
#     except np.linalg.LinAlgError:
#         # fallback if not positive definite
#         Q_sqrt = np.linalg.matrix_power(Q, 1 // 2)
#
#     # Check each eigenvalue
#     for lam in eigvals:
#         if abs(lam) >= 1:  # unstable mode
#             obs_matrix = np.vstack([lam * np.eye(n) - A, C])
#             if matrix_rank(obs_matrix) < n:
#                 stationary = False
#                 info['unstable_unobservable'].append(lam)
#             else:
#                 info['unstable_observable'].append(lam)
#
#     # Check stabilizability (controllability by Q)
#     controllability_matrix = Q_sqrt
#     for i in range(1, n):
#         controllability_matrix = np.hstack([controllability_matrix, np.linalg.matrix_power(A, i) @ Q_sqrt])
#     if matrix_rank(controllability_matrix) < n:
#         stationary = False
#         info['stabilizable'] = False
#
#     # Check R > 0
#     if not np.all(np.linalg.eigvals(R) > tol):
#         stationary = False
#         info['R_positive_definite'] = False
#     else:
#         info['R_positive_definite'] = True
#
#     return stationary, info

@njit
def _calc_forecasts(k, mu, Z, H, F, Q, x0, P0):
    dim = x0.shape[0]


    x_predictions = np.zeros((k, dim))
    P_state_predictions = np.zeros((k, dim, dim))

    N = Z.shape[0]
    y_predictions = np.zeros((k, N))
    P_obs_predictions = np.zeros((k, N, N))

    x_predictions[0] = F @ x0
    P_state_predictions[0] = F @ P0 @ F.T + Q
    y_predictions[0] = mu + Z @ x_predictions[0]
    P_obs_predictions[0] = Z @ P_state_predictions[0] @ Z.T + H

    for i in range(1, k):
        x_predictions[i] = F @ x_predictions[i - 1]
        P_state_predictions[i] = F @ P_state_predictions[i - 1] @ F.T + Q
        y_predictions[i] = mu + Z @ x_predictions[i]
        P_obs_predictions[i] = Z @ P_state_predictions[i] @ Z.T + H

    return y_predictions, P_obs_predictions, x_predictions, P_state_predictions


def simultaneous_congruence(F1, F2, tol=1e-10):
    """
    Diagonalize F1 and F2 jointly under congruence transformation.

    Returns:
        J1.T @ F1 @ J1 = I,
        J2.T @ F1 @ J2 = 0,
        J1.T @ F1 @ J2 = 0,
        J2.T @ F2 @ J2 = I,
        J1.T @ F2 @ J2 = 0,
        etc.
    """
    # Step 1: Symmetrize for numerical stability
    F1 = (F1 + F1.T) / 2
    F2 = (F2 + F2.T) / 2

    # Step 2: Eigendecomposition of F1
    eigvals_F1, U = eigh(F1)

    keep = eigvals_F1 > tol
    null = eigvals_F1 <= tol

    if not np.any(keep):
        return None, None

    U1 = U[:, keep]  # Image basis
    U2 = U[:, null]  # Nullspace basis

    # Step 3: Construct J1 (whiten F1 in its image)
    J1 = U1 @ np.diag(1.0 / np.sqrt(eigvals_F1[keep]))

    # Step 4: F2-orthogonalize U2 against U1
    F21 = U1.T @ F2 @ U2  # (r x m) matrix
    U2_proj = U2 - U1 @ F21  # subtract projection onto image under F2

    # Step 5: QR to re-orthonormalize U2_proj
    Q2, R = qr(U2_proj, mode='economic')
    # Some columns may be numerically zero, remove them
    nonzero = np.abs(np.diag(R)) > tol
    Q2 = Q2[:, nonzero]

    # Step 6: Project F2 into this cleaned subspace and whiten
    F2_proj = Q2.T @ F2 @ Q2
    eigvals_F2_proj, W = eigh(F2_proj)
    if np.any(eigvals_F2_proj < tol):
        eigvals_F2_proj[eigvals_F2_proj < tol] = tol
        # raise LinAlgError("F2 projection in nullspace is not positive definite")

    J2 = Q2 @ W @ np.diag(1.0 / np.sqrt(eigvals_F2_proj))

    return J1, J2

@njit
def calc_EM_matrices(endog, mu, x_smooth, P_smooth, Plag_smooth, burn=10):
    demeaned_endog = endog - mu

    T = x_smooth.shape[0]
    dim = x_smooth.shape[1]

    F = np.zeros((dim, dim))
    for t in range(burn, T):
        F += np.outer(x_smooth[t], x_smooth[t]) + P_smooth[t]

    C = F - np.outer(x_smooth[T - 1], x_smooth[T - 1]) - P_smooth[T - 1]
    A = F - np.outer(x_smooth[burn], x_smooth[burn]) - P_smooth[burn]

    B = np.outer(x_smooth[burn + 1], x_smooth[burn]) + Plag_smooth[burn + 1]
    for t in range(burn + 2, T):
        B += np.outer(x_smooth[t], x_smooth[t - 1]) + Plag_smooth[t]

    D = np.outer(demeaned_endog[burn], demeaned_endog[burn])
    for t in range(burn + 1, T):
        D += np.outer(demeaned_endog[t], demeaned_endog[t])

    E = np.outer(demeaned_endog[burn], x_smooth[burn])
    for t in range(burn + 1, T):
        E += np.outer(demeaned_endog[t], x_smooth[t])

    return A, B, C, D, E, F


def calc_Q(Aj, Bj, Cj, Dj, Ej, Fj, Z, H, F, Q, T):
    sum1 = - 0.5 * np.trace(Q @ (Aj - F @ Bj.T - Bj @ F.T + F @ Cj @ F.T))
    sum2 = - 0.5 * np.trace(np.linalg.inv(H) @ (Dj - Z @ Ej.T - Ej @ Z.T + Z @ Fj @ Z.T))
    return sum1 - (T / 2) * np.log(np.linalg.det(H)) + sum2


class KalmanFilterInitialization:

    _initialization_type_map = {"ss": "ss",
                                "steady_state": "ss",
                                "steadystate": "ss",
                                "ed": "ed",
                                "exact_diffuse": "ed",
                                "exactdiffuse": "ed",
                                "s": "s",
                                "specified": "s"}

    def __init__(self,
                 initialization_type: Literal[
                     "ss", "steady_state", "steadystate", "exact_diffuse", "exactdiffuse", "ed", "specified", "s"],
                 x0: np.ndarray,
                 P0: Optional[np.ndarray] = None,
                 P_star: Optional[np.ndarray] = None,
                 P_infty: Optional[np.ndarray] = None):

        if initialization_type not in self._initialization_type_map.keys():
            raise ValueError(f"Invalid initialization_type. Got {initialization_type}")

        self._initialization_type = self._initialization_type_map[initialization_type.lower()]
        self._x0 = x0
        self._P0 = P0
        self._P_infty = P_infty
        self._P_star = P_star

        self._check_initialize(initialization_type, x0, P0, P_star, P_infty)

    def _check_initialize(self, init_type, x0, P0, P_star, P_infty):

        if isinstance(x0, np.ndarray):
            if x0.ndim > 1:
                raise ValueError("x0 must be 1D")
        else:
            raise ValueError("x0 must be np.array")

        if not init_type in self._initialization_type_map.keys():
            raise ValueError(f"Invalid initialization_type. Got {init_type}")

        if self._initialization_type_map[init_type] == "ss":
            if all(m is None for m in [P0, P_star, P_infty]):
                return

        elif self._initialization_type_map[init_type] == "s":
            if all(m is None for m in [P_star, P_infty]) and isinstance(P0, np.ndarray):
                if P0.ndim == 2 and P0.shape[0] == P0.shape[1]:
                    if x0.shape[0] == P0.shape[0]:
                        return

        elif self._initialization_type_map[init_type] == "ed":
            if all(isinstance(m, np.ndarray) for m in [P_star, P_infty]) and P0 is None:
                if P_star.ndim == 2 and P_infty.ndim == 2:
                    if all((m.shape[0] == x0.shape[0] and m.shape[1] == x0.shape[0]) for m in [P_star, P_infty]):
                        return

        raise ValueError("Error initializing kalman filter. Check inputs")

    @property
    def type(self):
        return self._initialization_type

    @property
    def x0(self):
        return self._x0

    @property
    def P0(self):
        return self._P0

    @property
    def P_star(self):
        return self._P_star

    @property
    def P_infty(self):
        return self._P_infty


class KalmanFilterInSamplePrediction:

    def __init__(self,
                 prediction_errors: np.ndarray,
                 predicted_state: np.ndarray,
                 predicted_state_cov: np.ndarray,
                 nll: float,
                 prediction_horizon: int
                 ):

        self._prediction_errors = prediction_errors
        self._predicted_state = predicted_state
        self._predicted_state_cov = predicted_state_cov
        self._nll = nll
        self._prediction_horizon = prediction_horizon

    @property
    def prediction_errors(self):
        return self._prediction_errors

    @property
    def predicted_state(self):
        return self._predicted_state

    @property
    def predicted_state_cov(self):
        return self._predicted_state_cov

    @property
    def prediction_horizon(self):
        return self._prediction_horizon

    @property
    def nll(self):
        return self._nll


class KalmanFilterForecast:

    def __init__(self,
                 prediction_obs: np.ndarray,
                 prediction_obs_cov: np.ndarray,
                 predicted_state: np.ndarray,
                 predicted_state_cov: np.ndarray,
                 forecast_horizon: int,
                 M: Optional[np.ndarray] = None,):

        self._prediction_obs = prediction_obs
        self._prediction_obs_cov = prediction_obs_cov
        self._predicted_state = predicted_state
        self._predicted_state_cov = predicted_state_cov
        self._forecast_horizon = forecast_horizon

        self._M = M

    @property
    def forecasted_obs(self):
        return self._prediction_obs

    @property
    def forecasted_obs_cov(self):
        return self._prediction_obs_cov

    @property
    def forecasted_state(self):
        return self._predicted_state

    @property
    def forecasted_state_cov(self):
        return self._predicted_state_cov

    @property
    def forecasted_factors(self):
        if self._M is None:
            raise ValueError("M is None. No factors")
        return self.forecasted_state @ self._M.T

    @property
    def forecast_horizon(self):
        return self._forecast_horizon


class KalmanFilterResult(KalmanFilterInSamplePrediction):

    def __init__(self,
                 series: np.ndarray,
                 representation: LinearStateSpaceModelRepresentation,
                 initialization: KalmanFilterInitialization,
                 prediction_errors: np.ndarray,
                 predicted_state: np.ndarray,
                 predicted_state_cov: np.ndarray,
                 nll: float,
                 ZTSinv: np.ndarray
                 ):

        super().__init__(prediction_errors=prediction_errors,
                         predicted_state=predicted_state,
                         predicted_state_cov=predicted_state_cov,
                         nll=nll,
                         prediction_horizon=1)

        self._series = series
        self._representation = representation
        self._initialization = initialization

        self._ZTinvS = ZTSinv

        self._updated = None
        self._smoothed = None

    @property
    def series(self):
        return self._series

    @property
    def representation(self):
        return self._representation

    @property
    def initialization(self):
        return self._initialization

    @property
    def smoothed_state(self):
        if self._smoothed is None:
            self._smoothed = self._calc_smoothed()

        return self._smoothed[0]

    @property
    def smoothed_state_cov(self):
        if self._smoothed is None:
            self._smoothed = self._calc_smoothed()

        return self._smoothed[1]

    @property
    def updated_state(self):
        if self._updated is None:
            self._updated = self._calc_updated()

        return self._updated[0]

    @property
    def updated_state_cov(self):
        if self._updated is None:
            self._updated = self._calc_updated()

        return self._updated[1]

    @cached_property
    def last_updated_state(self):
        kalman_gain = self.predicted_state_cov[-1] @ self._ZTinvS[-1]
        return self.predicted_state[-1] + kalman_gain @ self.prediction_errors[-1]

    @cached_property
    def last_updated_state_cov(self):
        d = self.representation.F.shape[0]  # Dimension of state
        I_d = np.eye(d)
        kalman_gain = self.predicted_state_cov[-1] @ self._ZTinvS[-1]
        return (I_d - kalman_gain @ self.representation.Z) @ self.predicted_state_cov[-1]

    @cached_property
    def last_updated_state_and_cov(self):
        kalman_gain = self.predicted_state_cov[-1] @ self._ZTinvS[-1]
        state = self.predicted_state[-1] + kalman_gain @ self.prediction_errors[-1]
        I_d = np.eye(self.representation.F.shape[0])
        cov = (I_d - kalman_gain @ self.representation.Z) @ self.predicted_state_cov[-1]
        return state, cov

    def get_state(self, conditional_on: Literal["predicted", "updated", "smoothed"] = "smoothed"):
        if conditional_on == "predicted":
            return self.predicted_state
        elif conditional_on == "updated":
            return self.updated_state
        elif conditional_on == "smoothed":
            return self.smoothed_state
        else:
            raise ValueError("invalid input conditionalOn")

    def get_state_cov(self, conditional_on: Literal["predicted", "updated", "smoothed"] = "smoothed"):
        if conditional_on == "predicted":
            return self.predicted_state_cov
        elif conditional_on == "updated":
            return self.updated_state_cov
        elif conditional_on == "smoothed":
            return self.smoothed_state_cov
        else:
            raise ValueError("invalid input conditionalOn")

    def get_factors(self, conditional_on: Literal["predicted", "updated", "smoothed"] = "smoothed"):
        if conditional_on == "predicted":
            return self.predicted_state @ self.representation.M.T
        elif conditional_on == "updated":
            return self.updated_state @ self.representation.M.T
        elif conditional_on == "smoothed":
            return self.smoothed_state @ self.representation.M.T
        else:
            raise ValueError("invalid input conditionalOn")

    def calc_k_step_predictions(self, k: int):  # TODO: not well optimized, or done properly in general
        # y, mu, Z, H, F, R,

        # x_t+1 = F x_t + e, e ~N(0, R @ R.T)
        # y_t   = mu + Z x_t + u, u ~N(0, H)

        x_predicted = self.predicted_state
        P_predicted = self.predicted_state_cov

        y = self.series

        F = self.representation.F
        Q = self.representation.Q
        mu = self.representation.const
        Z = self.representation.Z
        H = self.representation.H

        T, dim = x_predicted.shape
        N = y.shape[1] if y.ndim > 1 else 1

        # x_predicted   is x_{t|t-1}
        # x_predicted_k is x_{t|t-k} = F x_{t-1|t-k} = F F x_{t-2|t-k} = ... = F^(k-1) x_{t-(k-1)|t-(k-1)-1}
        #                            = F^(k-1) x_{m|m-1}, m = t-(k-1)

        x_predicted_k = np.zeros((T, dim))
        kAheadErrors = np.zeros((T, N))

        k_1 = k - 1
        F_to_K_1 = np.linalg.matrix_power(F, k_1)

        demeaned_endog = y - mu

        covIsSteadyState = isinstance(P_predicted, SteadyStateArray)
        if covIsSteadyState:
            mat = P_predicted[0].copy()
            for _ in range(k_1):
                mat = F @ mat @ F.T + Q
            P_predicted_k = SteadyStateArray(mat, T)
            predErrorCov = Z @ mat @ Z.T + H

            c, lower = cho_factor(predErrorCov[0], check_finite=False)
            logdet = 2 * np.sum(np.log(np.diag(c)))

            inv_cov = cho_solve((c, lower), np.eye(predErrorCov.shape[0]))
            predError_precision = SteadyStateArray(inv_cov, T)
        else:
            P_predicted_k = np.zeros((T, dim, dim))
            predError_precision = np.zeros((T, N, N))

            F_powers = [np.eye(dim)]
            for i in range(1, k):
                F_powers.append(F @ F_powers[-1])  # F^i

        log_likelihood = 0.0
        for t in range(k_1, T):
            x_predicted_k[t] = F_to_K_1 @ x_predicted[t - k_1]
            kAheadErrors[t] = demeaned_endog[t] - Z @ x_predicted_k[t]

            if not covIsSteadyState:
                P_k = F_powers[-1] @ P_predicted[t - k_1] @ F_powers[-1].T
                for i in range(k_1):
                    P_k += F_powers[i] @ Q @ F_powers[i].T
                P_predicted_k[t] = P_k

                predErrorCov = Z @ P_k @ Z.T + H
                c, lower = cho_factor(predErrorCov, check_finite=False)
                logdet = 2 * np.sum(np.log(np.diag(c)))

                predError_precision[t] = cho_solve((c, lower), np.eye(predErrorCov.shape[0]))

            quad_form = kAheadErrors[t].T @ cho_solve((c, lower), kAheadErrors[t])
            ll = -0.5 * (N * np.log(2 * np.pi) + logdet + quad_form)
            log_likelihood += ll

        res = KalmanFilterInSamplePrediction(kAheadErrors, x_predicted_k, P_predicted_k, -float(log_likelihood), k)

        return res

    def get_EM_matrices(self):
        if self._smoothed is None:
            self._smoothed = self._calc_smoothed()
        x_smooth, P_smooth, Plag_smooth = self._smoothed[0:3]
        return calc_EM_matrices(self.series, self.representation.const, x_smooth, P_smooth, Plag_smooth, burn=10)

    def _calc_smoothed(self):
        ZTinvS = self._ZTinvS
        predicted_state_cov = self.predicted_state_cov

        if isinstance(ZTinvS, SteadyStateArray):
            ZTinvS = ZTinvS.mat
            # ZTinvS = ZTinvS.to_array()
            # ZTinvS = np.ascontiguousarray(ZTinvS.to_array())
        elif isinstance(ZTinvS, PowerExpansionArray):
            ZTinvS = ZTinvS.base_matrices
        if isinstance(predicted_state_cov, SteadyStateArray) or isinstance(predicted_state_cov, PowerExpansionArray):
            predicted_state_cov = predicted_state_cov.mat
            # predicted_state_cov = predicted_state_cov.to_array()
            # predicted_state_cov = np.ascontiguousarray(predicted_state_cov.to_array())
        elif isinstance(predicted_state_cov, PowerExpansionArray):
            predicted_state_cov = predicted_state_cov.base_matrices

        return kalman_smoother(self.representation.Z, self.representation.F, self.prediction_errors,
                               self.predicted_state, predicted_state_cov, ZTinvS)

    def _calc_updated(self):
        return _kalman_updater(self)

    def forecast(self, k: int):
        mu, Z, H, F, Q = (self.representation.const, self.representation.Z, self.representation.H,
                          self.representation.F, self.representation.Q)
        x0, P0 = self.last_updated_state_and_cov
        y, Py, x, Px = _calc_forecasts(k, mu, Z, H, F, Q, x0, P0)

        return KalmanFilterForecast(y, Py, x, Px, k, self.representation.M)

    def stream(self, y: np.ndarray): # TODO
        raise NotImplementedError

    def prediction_error_and_cov(self): # TODO
        raise NotImplementedError

    def loss(self, loss: OptimizationObjective):
        if loss.requires_cov:
            raise NotImplementedError
        else:
            pred_error = self.prediction_errors
            return loss(pred_error)


class SteadyStateArray:

    def __init__(self, mat, T):
        self.mat = mat
        self.T = T

    def __getitem__(self, i):
        return np.broadcast_to(self.mat, self.shape)[i]

    @property
    def shape(self):
        return (self.T,) + self.mat.shape

    def to_array(self):
        d1, d2 = self.mat.shape
        return np.broadcast_to(self.mat, (self.T, d1, d2))

    def __array__(self, dtype=None):
        arr = np.broadcast_to(self.mat, self.shape)
        if dtype:
            arr = arr.astype(dtype)
        return arr


class PowerExpansionArray:

    def __init__(self, mat0, mat1, t_drop, scale=1e10):
        # mat = mats_0 + k mats_1, k -> infty

        self.initial_mat0 = mat0[0:t_drop].copy()
        self.initial_mat1 = mat1[0:t_drop]

        self.t_drop = t_drop

        self.mat = mat0
        for t in range(t_drop):
            self.mat[t] += scale * mat1[t]

    def __getitem__(self, i):
        return self.mat[i]

    def to_array(self):
        return self.mat

    @property
    def shape(self):
        return self.mat.shape

    @property
    def base_matrices(self):
        mat = self.mat.copy()
        mat[1:self.t_drop] = self.initial_mat0
        return mat

    @property
    def expansion_matrices(self):
        return self.initial_mat1

    def __array__(self, dtype=None):
        arr = self.mat.copy()
        if dtype:
            arr = arr.astype(dtype)
        return arr


class KalmanFilter:
    """
    Kalman filter for a linear Guassian state space model, with the form

    y_t = mu + Z x_t + e_t,  e_t ~ N(0, H)

    x_{t+1} = F x_t + R u_t,  u_t ~ N(0, Q)

    """

    _steady_state_calculation_opts = ["riccati", "lyapunov", "dare"]
    # Riccati and Lyapunov can be used with a warm start, good for mle procedures.
    # For non-stationary states an exact diffuse initialization should be used, but Riccati can provide good result
    #   if numerical convergence is achieved, even tho this is not principaled

    def __init__(self):
        self._endog = None

        self._mutable_representation: Optional[MutableLinearStateSpaceModelRepresentation] = None

        self._initialization: Optional[KalmanFilterInitialization] = None

        self._steady_state_calculation = "dare"

    @property
    def steady_state_calculation(self):
        return self._steady_state_calculation

    @steady_state_calculation.setter
    def steady_state_calculation(self, value):
        value = value.lower()
        if value not in self._steady_state_calculation_opts:
            raise ValueError(f"Invalid steady state calculation option: {value}. Options are {self._steady_state_calculation_opts}")
        self._steady_state_calculation = value

    def set_steady_state_calculation(self, value):
        self.steady_state_calculation = value
        return self

    @property
    def endog(self):
        return self._endog

    @endog.setter
    def endog(self, endog):
        self._endog = endog

    def set_endog(self, endog):
        self.endog = endog
        return self

    @property
    def representation(self) -> LinearStateSpaceModelRepresentation:
        if self._mutable_representation is None:
            raise RuntimeError("Representation has not been set.")
        return self._mutable_representation.get_frozen()

    @representation.setter
    def representation(self, representation: LinearStateSpaceModelRepresentation):
        self._representation_setter(representation, True)

    def set_representation(self, representation: LinearStateSpaceModelRepresentation):
        self.representation = representation
        return self

    def set_representation_trusted(self, representation: LinearStateSpaceModelRepresentation):
        self._representation_setter(representation, False)
        return self

    def _representation_setter(self,
                               representation: LinearStateSpaceModelRepresentation,
                               validate: bool = True):

        if self._mutable_representation is None:
            self._mutable_representation = MutableLinearStateSpaceModelRepresentation.from_frozen(representation, validate)
        else:
            self._mutable_representation.update_representation(representation, validate)

        self._mutable_representation.steady_state_calculation_method = self.steady_state_calculation

    @property
    def initialization(self):
        return self._initialization

    @initialization.setter
    def initialization(self, initialization: KalmanFilterInitialization):
        self._initialization = initialization

    def set_initialization(self, initialization: KalmanFilterInitialization):
        self.initialization = initialization
        return self

    def filter(self):
        pred_errors, x_predicted, P_predicted, nll, ZTSinv = self._calc_filtered()
        res = KalmanFilterResult(self.endog, self.representation, self.initialization,
                                 pred_errors, x_predicted, P_predicted, nll, ZTSinv)

        return res

    # def filter_with_dynamax(self):
    #     self._check_properly_defined()
    #
    #     endog = self.endog
    #     rep = self.representation
    #     const, Z, H, F, RQRT = rep.const, rep.E @ rep.M, rep.H, rep.F, rep.RQRT
    #
    #     x0 = self.initialization.x0
    #
    #     if self.initialization.type == "ss":
    #         P0 = self.representation.steady_state_covariance
    #     elif self.initialization.type == "s":
    #         P0 = self.initialization.P0
    #     elif self.initialization.type == "ed":
    #         P0 = self.initialization.P_star + 1e8 * self.initialization.P_infty
    #     else:
    #         raise ValueError("Error initializing kalman filter. Check inputs")
    #
    #     state_dim = F.shape[0]
    #     emission_dim = Z.shape[0]
    #
    #     _initial_mean = jax.device_put(x0),
    #     _initial_covariance = jax.device_put(P0)
    #     _dynamics_weights = jax.device_put(F)
    #     _dynamics_input_weights = jnp.zeros((state_dim, 1))
    #     _dynamics_bias = jnp.zeros((state_dim,))
    #     _dynamics_covariance = jax.device_put(RQRT)
    #     _emission_weights = jax.device_put(Z)
    #     _emission_input_weights = jnp.zeros((emission_dim, 1))
    #     _emission_bias = jax.device_put(const)
    #     _emission_covariance = jax.device_put(H)
    #
    #
    #     params = ParamsLGSSM(
    #         initial=ParamsLGSSMInitial(
    #             mean=_initial_mean,
    #             cov=_initial_covariance),
    #         dynamics=ParamsLGSSMDynamics(
    #             weights=_dynamics_weights,
    #             bias=_dynamics_bias,
    #             input_weights=_dynamics_input_weights,
    #             cov= _dynamics_covariance),
    #         emissions=ParamsLGSSMEmissions(
    #             weights=_emission_weights,
    #             bias=_emission_bias,
    #             input_weights=_emission_input_weights,
    #             cov=_emission_covariance)
    #     )
    #
    #     return lgssm_filter(params, jax.device_put(endog))

    def _calc_filtered(self):

        self._check_properly_defined()

        endog = self.endog
        rep = self.representation
        const, Z, H, F, RQRT = rep.const, rep.E @ rep.M, rep.H, rep.F, rep.RQRT

        x0 = self.initialization.x0

        if self.initialization.type == "ss":
            ss_cov = self.representation.steady_state_covariance
            ss_kg = self.representation.steady_state_kalman_gain
            ss_obs_prec = self.representation.steady_state_observation_precision

            nll, pred_errors, x_predicted = \
                kalman_filter_SS(endog, const, Z, H, F, x0, ss_cov, ss_kg, ss_obs_prec)

            P_predicted = SteadyStateArray(ss_cov, self.endog.shape[0])
            ZTSinv = SteadyStateArray(Z.T @ ss_obs_prec, self.endog.shape[0])

        elif self.initialization.type == "s":
            nll, pred_errors, x_predicted, P_predicted, ZTSinv = \
                kalman_filter(endog, const, Z, H, F, RQRT, x0, self.initialization.P0)

        elif self.initialization.type == "ed":

            nll, pred_errors, x_predicted, P_predicted, ZTSinv = \
                kalman_filter_exact_diffuse(endog, const, Z, H, F, RQRT, x0,
                                            self.initialization.P_star, self.initialization.P_infty)

        else:
            raise ValueError("Error initializing kalman filter. Check inputs")

        return pred_errors, x_predicted, P_predicted, nll, ZTSinv

    def nll(self):

        self._check_properly_defined()

        endog = self.endog
        rep = self.representation
        const, Z, H, F, RQRT = rep.const, rep.E @ rep.M, rep.H, rep.F, rep.RQRT

        x0 = self.initialization.x0

        if self.initialization.type == "ss":
            ss_cov = self.representation.steady_state_covariance
            ss_kg = self.representation.steady_state_kalman_gain
            ss_obs_prec = self.representation.steady_state_observation_precision

            nll = kalman_nll_SS(endog - const, F, Z, H,
                                ss_kg, ss_obs_prec, ss_cov, x0)

        elif self.initialization.type == "s":
            nll, _, _, _, _ = \
                kalman_filter(endog, const, Z, H, F, RQRT, x0, self.initialization.P0)

        elif self.initialization.type == "ed":
            nll, _, _, _, _ = \
                kalman_filter_exact_diffuse(endog, const, Z, H, F, RQRT, x0,
                                            self.initialization.P_star, self.initialization.P_infty)

        else:
            raise ValueError("Error initializing kalman filter. Check inputs")

        return nll

    def k_ahead_nll(self, k: int):  # TODO: not well optimized
        if k == 1:
            return self.nll()
        self._check_properly_defined()
        kf_result = self.filter()
        kf_pred_result = kf_result.calc_k_step_predictions(k)
        return kf_pred_result.nll

    def css(self):
        pred_error = self.get_prediction_error()
        return np.sum(pred_error ** 2)

    def k_ahead_css(self, k: int):  # TODO: not well optimized
        if k == 1:
            return self.css()
        self._check_properly_defined()
        kf_result = self.filter()
        ss_pred_result = kf_result.calc_k_step_predictions(k)
        return np.sum(ss_pred_result.prediction_errors ** 2)

    def get_prediction_error(self):
        self._check_properly_defined()

        endog = self.endog
        rep = self.representation
        const, Z, H, F, RQRT = rep.const, rep.E @ rep.M, rep.H, rep.F, rep.RQRT

        x0 = self.initialization.x0

        if self.initialization.type == "ss":
            ss_kg = self.representation.steady_state_kalman_gain

            pred_error = kalman_pred_error_SS(endog - const, F, Z, ss_kg, x0)

        elif self.initialization.type == "s":
            _, pred_error, _, _, _ = \
                kalman_filter(endog, const, Z, H, F, RQRT, x0, self.initialization.P0)

        elif self.initialization.type == "ed":
            _, pred_error, _, _, _, _ = \
                kalman_filter_exact_diffuse(endog, const, Z, H, F, RQRT, x0,
                                            self.initialization.P_star, self.initialization.P_infty)

        else:
            raise ValueError("Error initializing kalman filter. Check inputs")

        return pred_error

    def get_prediction_error_and_cov(self):
        raise NotImplementedError

    def loss(self, loss: OptimizationObjective):
        if loss.requires_cov:
            if isinstance(loss, GaussianNLL):
                return self.nll()
            raise NotImplementedError("Currently, the kalman filter only suppoers NLL loss and losses based on prediction errors")
        else:
            pred_error = self.get_prediction_error()
            return loss(pred_error)

    def k_ahead_loss(self, k: int, loss: OptimizationObjective):
        if k == 1:
            return self.loss(loss)
        else:
            if loss.requires_cov:
                raise NotImplementedError
            else:
                self._check_properly_defined()
                kf_result = self.filter()
                ss_pred_result = kf_result.calc_k_step_predictions(k)
                return loss(ss_pred_result.prediction_errors)

    def _check_properly_defined(self):

        if not isinstance(self.initialization, KalmanFilterInitialization):
            raise ValueError("kalman filter must be initialized before calling filter")

        if not isinstance(self._mutable_representation, MutableLinearStateSpaceModelRepresentation):
            raise ValueError("representation needs to be set before calling filter")

        self._check_dims()

    def _check_dims(self):

        if self.endog.ndim > 1:
            n_obs_endog = self.endog.shape[1]
        else:
            n_obs_endog = 1

        n_obs_mu = self._mutable_representation.const.size
        Z = self._mutable_representation.E @ self._mutable_representation.M

        n_obs_Z = Z.shape[0]
        n_obs_H1 = self._mutable_representation.H.shape[0]
        n_obs_H2 = self._mutable_representation.H.shape[1]

        n_state_Z = Z.shape[1]
        n_state_F1 = self._mutable_representation.F.shape[0]
        n_state_F2 = self._mutable_representation.F.shape[1]
        n_state_R = self._mutable_representation.R.shape[0]

        all_obs = [n_obs_endog, n_obs_mu, n_obs_Z, n_obs_H1, n_obs_H2]
        if len(set(all_obs)) > 1:
            raise ValueError("Dimension miss match: observation side")

        all_state = [n_state_Z, n_state_F1, n_state_F2, n_state_R]
        if len(set(all_state)) > 1:
            raise ValueError("Dimension miss match: State side")

    def get_em_matrices(self, burn: int = 10):
        self._check_properly_defined()

        pred_errors, x_predicted, predicted_state_cov, nll, ZTinvS = self._calc_filtered()

        if isinstance(ZTinvS, SteadyStateArray):
            ZTinvS = ZTinvS.mat
        elif isinstance(ZTinvS, PowerExpansionArray):
            ZTinvS = ZTinvS.base_matrices
        if isinstance(predicted_state_cov, SteadyStateArray) or isinstance(predicted_state_cov, PowerExpansionArray):
            predicted_state_cov = predicted_state_cov.mat
        elif isinstance(predicted_state_cov, PowerExpansionArray):
            predicted_state_cov = predicted_state_cov.base_matrices

        Z, F = self.representation.E @ self.representation.M, self.representation.F

        x_smooth, P_smooth, Plag_smooth = kalman_smoother(Z, F, pred_errors, x_predicted, predicted_state_cov, ZTinvS)

        return calc_EM_matrices(self.endog, self.representation.const, x_smooth, P_smooth, Plag_smooth, burn=burn)
