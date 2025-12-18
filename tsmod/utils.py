import numpy as np

def chol_cov(series):
    cov = np.cov(series)
    return np.linalg.cholesky(cov)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def softplus_inv(y):
    # return y + np.log1p(-np.exp(-y))
    # if np.any(y <= 0):
    #     raise ValueError("softplus_inv(y) is only defined for y > 0")

    thresh = 20.0  # threshold to switch formulas
    return np.where(
        y > thresh,
        y + np.log1p(-np.exp(-y)),  # large y
        np.log(np.expm1(y))  # small y
    )


def sigmoid(x):
    # derivative of softplus
    # return 1/(1+np.exp(-x))
    return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))


def sigmoid_softplus_inv(x):
    return 1 - np.exp(-x)


def numerical_jacobian(f, x, epsilon=1e-8):
    """
    Compute the numerical Jacobian of a vector-valued function f at point x.

    Parameters:
        f       : callable, function mapping from R^n to R^m
        x       : np.ndarray, shape (n,), input point to evaluate Jacobian at
        epsilon : float, small perturbation for finite difference

    Returns:
        J       : np.ndarray, shape (m,n), numerical Jacobian matrix
    """
    x = x.astype(float)  # ensure float type
    n = x.shape[0]
    f_x = f(x)
    m = f_x.shape[0]
    J = np.zeros((m, n))

    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += epsilon
        f_x_perturbed = f(x_perturbed)
        J[:, i] = (f_x_perturbed - f_x) / epsilon

    return J


def permutation_matrix_column_to_row_major(n, m):

    P = np.zeros((m*n, m*n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(m):
            P[idx, i + j*n] = True
            idx += 1

    return P


def indexing_column_to_row_major(n, m):

    # idxs = np.empty((m*n, ), dtype=int)
    # idx = 0
    # for i in range(n):
    #     for j in range(m):
    #         idxs[idx] = i + j*n
    #         idx += 1
    # return idxs

    i = np.arange(n)  # Array of row indices
    j = np.arange(m)  # Array of column indices
    idxs = i[:, None] + j * n  # Broadcasting to create the desired index pattern

    return idxs.flatten('C')


def indexing_row_to_column_major(n, m):
    # idxs = np.empty((m*n, ), dtype=int)
    # idx = 0
    # for i in range(n):
    #     for j in range(m):
    #         idxs[i + j*n] = idx
    #         idx += 1
    # return idxs

    i = np.arange(n)[:, None]  # shape (n,1)
    j = np.arange(m)  # shape (m,)
    idxs = i * m + j  # row-major indices
    return idxs.flatten('F')  # transpose to match column-major order


