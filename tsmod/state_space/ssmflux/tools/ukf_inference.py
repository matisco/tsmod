import numpy as np
from numba import njit, float64, int64, types

@njit(float64[:, :](float64[:], float64[:, :], float64), cache=True)
def generate_sigma_points(x: np.ndarray,
                          P: np.ndarray,
                          lambda_: float):
    """
    Generate sigma points and weights for the Unscented Kalman Filter.

    Parameters:
    -----------
    x : ndarray, shape (n,)
        State mean vector
    P : ndarray, shape (n, n)
        State covariance matrix
    alpha : float
        Spread of sigma points (usually small, e.g., 1e-3)
    kappa : float
        Secondary scaling parameter (usually 0 or 3-n)

    Returns:
    --------
    sigma_points : ndarray, shape (2n+1, n)
        Generated sigma points
    """
    dim = x.shape[0]
    # lambda_ = alpha**2 * (dim + kappa) - dim

    sqrt_P = np.linalg.cholesky((dim + lambda_) * P)

    sigma_points = np.zeros((2 * dim + 1, dim))
    sigma_points[0] = x
    for i in range(dim):
        sigma_points[i + 1]       = x + sqrt_P[:, i]
        sigma_points[dim + i + 1] = x - sqrt_P[:, i]

    return sigma_points


@njit(types.Tuple((float64[:], float64[:]))(int64, float64, float64, float64), cache=True)
def compute_weights(dim, alpha, beta, lambda_):

    # lambda_ = alpha**2 * (dim + kappa) - dim
    factor = 1 / (2 * (dim + lambda_))

    Wm = np.full(2 * dim + 1, factor)
    Wc = np.full(2 * dim + 1, factor)
    Wm[0] = lambda_ / (dim + lambda_)
    Wc[0] = lambda_ / (dim + lambda_) + (1 - alpha ** 2 + beta)
    return Wm, Wc


@njit(    types.Tuple((float64[:], float64[:, :]))(
    float64[:],
    float64[:, :],
    types.FunctionType(float64[:](float64[:], float64[:])),
    float64[:, :],
    float64[:],
    float64,
    float64[:],
    float64[:]),
    cache=True)
def ukf_predict(x, P, f, Q, u, lambda_, weights_mean, weights_cov):
    """
    Unscented Kalman Filter prediction step.

    Parameters:
    -----------
    x : ndarray, shape (n,)
        State mean
    P : ndarray, shape (n, n)
        State covariance
    f : callable
        State transition function: x_next = f(x, u)
    Q : ndarray, shape (n, n)
        Process noise covariance
    u : ndarray, optional
        Control input
    alpha, kappa : float
        UKF parameters

    Returns:
    --------
    x_pred : ndarray, shape (n,)
        Predicted state mean
    P_pred : ndarray, shape (n, n)
        Predicted state covariance
    """
    n = x.shape[0]

    sigma_points = generate_sigma_points(x, P, lambda_)

    sigma_points_pred = np.zeros_like(sigma_points)
    for i, sp in enumerate(sigma_points):
        sigma_points_pred[i] = f(sp, u)

    x_pred = np.sum(weights_mean[:, None] * sigma_points_pred, axis=0)

    P_pred = Q.copy()
    for i in range(2 * n + 1):
        y = sigma_points_pred[i] - x_pred
        P_pred += weights_cov[i] * np.outer(y, y)

    return x_pred, P_pred


@njit(    types.Tuple((float64[:], float64[:, :], float64[:, :]))(
    float64[:],
    float64[:, :],
    types.FunctionType(float64[:](float64[:], float64[:])),
    float64[:, :],
    float64[:],
    float64,
    float64[:],
    float64[:]),
    cache=True)
def ukf_predict_obs(x, P, z, H, u, lambda_, weights_mean, weights_cov):
    n = x.shape[0]
    obs_dim = H.shape[0]

    sigma_points = generate_sigma_points(x, P, lambda_)

    sigma_points_obs_pred = np.zeros((sigma_points.shape[0], obs_dim))
    for i, sp in enumerate(sigma_points):
        sigma_points_obs_pred[i] = z(sp, u)

    obs_pred = np.sum(weights_mean[:, None] * sigma_points_obs_pred, axis=0)
    obs_cov = np.ascontiguousarray(H.copy())
    P_x_obs_cross = np.zeros((n, obs_dim))
    for i in range(2 * n + 1):
        y = sigma_points_obs_pred[i] - obs_pred
        obs_cov += weights_cov[i] * np.outer(y, y)
        P_x_obs_cross += weights_cov[i] * np.outer(sigma_points[i] - x, y)

    obs_cov = 0.5 * (obs_cov + obs_cov.T)

    return obs_pred, obs_cov, P_x_obs_cross


@njit(    types.Tuple((float64[:], float64[:, :]))(float64[:], float64[:, :], float64[:], float64[:, :], float64[:, :], float64[:]),
    cache=True)
def ukf_update(x, P, obs_pred, obs_cov, P_x_obs_cross, true_value):
    obs_cov = np.ascontiguousarray(obs_cov)

    L = np.linalg.cholesky(obs_cov)
    y = np.linalg.solve(L, P_x_obs_cross.T)
    K = np.linalg.solve(L.T, y).T
    m_cond = x + K @ (true_value - obs_pred)
    P_cond = P - K @ obs_cov @ K.T

    return m_cond, P_cond


@njit(    float64(float64[:], float64[:, :], float64[:]), cache=True)
def calc_nll(mean, cov, value):
    L = np.linalg.cholesky(cov)

    pred_error = value - mean
    y = np.linalg.solve(L, pred_error)
    quad_form = np.dot(y, y)

    logdet = 2 * np.sum(np.log(np.diag(L)))

    N = mean.shape[0]
    return (logdet + quad_form + N * np.log(2 * np.pi))/2

@njit(    types.Tuple((float64, float64[:, :], float64[:, :, :], float64[:, :], float64[:, :, :]))(
    types.FunctionType(float64[:](float64[:], float64[:])),
    float64[:, :],
    types.FunctionType(float64[:](float64[:], float64[:])),
    float64[:, :],
    float64[:, :],
    float64[:, :],
    float64[:],
    float64[:, :],
    float64,
    float64,
    float64,
    float64[:, :]),
    cache=True)
def ukf_filter(f, Q, z, H, u_dynamics, u_observation, prior_x, prior_P, alpha, beta, kappa, observations):
    Q = np.ascontiguousarray(Q)
    H = np.ascontiguousarray(H)
    u_dynamics = np.ascontiguousarray(u_dynamics)
    u_observation = np.ascontiguousarray(u_observation)
    prior_x = np.ascontiguousarray(prior_x)
    prior_P = np.ascontiguousarray(prior_P)
    observations = np.ascontiguousarray(observations)

    T, obs_dim = observations.shape

    state_dim = prior_x.shape[0]

    lambda_ = alpha ** 2 * (state_dim + kappa) - state_dim

    weights_mean, weights_cov = compute_weights(state_dim, alpha, beta, lambda_)

    x_predicted = np.zeros((T, state_dim))
    P_predicted = np.zeros((T, state_dim, state_dim))
    obs_pred_error = np.zeros((T, obs_dim))
    obs_cov_predicted = np.zeros((T, obs_dim, obs_dim))
    tot_nll = 0

    for t in range(T):

        x_pred, P_pred = ukf_predict(prior_x, prior_P, f, Q, u_dynamics[t], lambda_, weights_mean, weights_cov)
        x_predicted[t] = x_pred
        P_predicted[t] = P_pred

        obs_pred, obs_cov, P_x_obs_cross = ukf_predict_obs(x_pred, P_pred, z, H, u_observation[t], lambda_, weights_mean, weights_cov)

        pred_error = observations[t] - obs_pred

        obs_pred_error[t] = pred_error
        obs_cov_predicted[t] = obs_cov

        prior_x, prior_P = ukf_update(x_pred, P_pred, obs_pred, obs_cov, P_x_obs_cross, observations[t])

        tot_nll += calc_nll(obs_pred, obs_cov, observations[t])

    return tot_nll, obs_pred_error, obs_cov_predicted, x_predicted, P_predicted


# @njit(cache=True)
# def ukf_filter_all(f, Q, z, H, u_dynamics, u_observation, prior_x, prior_P, alpha, beta, kappa, observations):
#
#     T, obs_dim = observations.shape
#
#     state_dim = prior_x.shape[0]
#
#     lambda_ = alpha ** 2 * (state_dim + kappa) - state_dim
#
#     weights_mean, weights_cov = compute_weights(state_dim, alpha, beta, lambda_)
#
#     x_predicted = np.zeros((T, state_dim))
#     P_predicted = np.zeros((T, state_dim, state_dim))
#     obs_pred_error = np.zeros((T, obs_dim))
#     obs_cov_predicted = np.zeros((T, obs_dim, obs_dim))
#     kalman_gain = np.zeros((T, state_dim, obs_dim))
#     nll = 0
#
#     for t in range(T):
#
#         # --------
#         # state prediction
#         # -------
#         sigma_points = generate_sigma_points(prior_x, prior_P, lambda_)
#
#         sigma_points_pred = np.zeros_like(sigma_points)
#         for i, sp in enumerate(sigma_points):
#             sigma_points_pred[i] = f(sp, u_dynamics[t])
#         x_pred = np.sum(weights_mean[:, None] * sigma_points_pred, axis=0)
#
#         P_pred = Q.copy()
#         for i in range(2 * state_dim + 1):
#             y = sigma_points_pred[i] - x_pred
#             P_pred += weights_cov[i] * np.outer(y, y)
#
#         x_predicted[t] = x_pred
#         P_predicted[t] = P_pred
#
#         # --------
#         # obs prediction
#         # -------
#
#         sigma_points = generate_sigma_points(x_pred, P_pred, lambda_)
#
#         sigma_points_obs_pred = np.zeros((sigma_points.shape[0], obs_dim))
#         for i, sp in enumerate(sigma_points):
#             sigma_points_obs_pred[i] = z(sp, u_observation[t])
#
#         obs_pred = np.sum(weights_mean[:, None] * sigma_points_obs_pred, axis=0)
#         obs_cov = H.copy()
#         P_x_obs_cross = np.zeros((state_dim, obs_dim))
#         for i in range(2 * state_dim + 1):
#             y = sigma_points_obs_pred[i] - obs_pred
#             obs_cov += weights_cov[i] * np.outer(y, y)
#             P_x_obs_cross += weights_cov[i] * np.outer(sigma_points[i] - x_pred, y)
#
#         pred_error = observations[t] - obs_pred
#
#         obs_pred_error[t] = pred_error
#         obs_cov_predicted[t] = obs_cov
#
#         # --------
#         # negative log-likelihood updated (without constants and factors)
#         # -------
#
#         L = np.linalg.cholesky(obs_cov)
#         y = np.linalg.solve(L, pred_error)
#         quad_form = np.dot(y, y)
#         logdet = 2 * np.sum(np.log(np.diag(L)))
#         nll += logdet + quad_form
#
#         # --------
#         # updating
#         # -------
#
#         L = np.linalg.cholesky((obs_cov + obs_cov.T) / 2)
#         y = np.linalg.solve(L, P_x_obs_cross.T)
#         K = np.linalg.solve(L.T, y).T
#         kalman_gain[t] = K
#
#         # current updated is next prior
#         prior_x = x_predicted[t] + K @ pred_error
#         prior_P = P_predicted[t] - K @ obs_cov @ K.T
#
#     nll = nll / 2
#     nll_constant = np.log(2 * np.pi) * obs_dim
#     nll += nll_constant * T / 2
#
#     return nll, obs_pred_error, obs_cov_predicted, x_predicted, P_predicted, kalman_gain

