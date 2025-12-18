import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space, expm
from scipy.stats import chi2

from numba import njit

from constrained_matrices import SpecialOrthogonalConstraint, SkewSymmetricConstraint


def calc_lagged_inner_product(x: np.ndarray, m: int) -> np.ndarray:
    """
    Compute lagged inner product matrices for a range of lags up to m.

    Parameters:
    x (np.ndarray): Input array of shape (n_samples, dim).
    m (int): Maximum lag.

    Returns:
    np.ndarray: 3D array of shape (dim, dim, m+1) containing sums of outer products at each lag.
    """
    T, dim = x.shape
    res = np.zeros((dim, dim, m + 1))

    for l in range(m + 1):
        if l == 0:
            product = x.T @ x
        else:
            product = x[l:].T @ x[:-l]
        res[:, :, l] = product / T

    return res


class MattesonAndTsay:

    @staticmethod
    def calc_rotation_matrix(factors: np.ndarray, max_lag: int, n_starting_vectors: int):
        n_factors = factors.shape[1]

        demeaned_factors = factors - np.mean(factors)

        Ss = calc_lagged_inner_product(demeaned_factors, max_lag)

        # rotation_matrix_wrapper = SpecialOrthogonalConstraint((n_factors, n_factors))
        skew_symmetric = SkewSymmetricConstraint((n_factors, n_factors))

        dof = skew_symmetric.n_params

        def obj(x):
            skew_symmetric.update_params(x)
            W = expm(skew_symmetric.matrix)

            obj = 0

            valid_idx = ~np.eye(n_factors, dtype=bool)

            l = 0
            current_weight = 2 * (1 - l / (max_lag + 1))
            current_S = W @ Ss[:, :, l] @ W.T
            obj += current_weight * np.sum(np.tril(current_S)[valid_idx.T] ** 2)  # tril picks lower triangle

            for l in range(1, max_lag + 1):
                current_weight = (1 - l / (max_lag + 1))
                current_S = W @ Ss[:, :, l] @ W.T
                obj += current_weight * np.sum(current_S[valid_idx] ** 2)

            return obj

        U = np.random.rand(dof, n_starting_vectors) * 2 * np.pi - np.pi
        fvals = np.zeros((n_starting_vectors,))
        xs = np.zeros((n_starting_vectors, dof))

        options = {'maxiter': 100, 'disp': False}

        for i in range(n_starting_vectors):
            x0 = U[:, i]
            res = minimize(obj, x0, method='SLSQP', options=options)  # SLSQP
            fvals[i] = res.fun
            xs[i, :] = res.x

        # Assuming x should be the one with the minimal function value
        best_index = np.argmin(fvals)
        best_x = xs[best_index, :]

        # rotation_matrix_wrapper.update_params(best_x)
        # return rotation_matrix_wrapper.matrix

        skew_symmetric.update_params(best_x)
        return expm(skew_symmetric.matrix)



if __name__ == '__main__':

    import numpy as np
    from scipy.signal import lfilter
    from matplotlib import pyplot as plt
    import time

    def generate_ar1(phi, n, sigma=1.0, x0=0.0):
        # white noise
        eps = np.random.normal(scale=sigma, size=n)

        # AR(1) filter coefficients: x_t = phi x_{t-1} + eps_t
        a = [1, -phi]  # AR polynomial
        b = [1]  # MA polynomial (just noise)

        # generate AR(1)
        x = lfilter(b, a, eps)

        # set initial condition if desired
        if x0 != 0:
            x = x + x0 - x[0]
        return x

    n_factors = 5
    T = 1000
    factors = np.empty((T, n_factors))

    for i in range(n_factors):
        factors[:, i] = generate_ar1(np.random.rand() * 0.5 + 0.4, T)

    rotation_wrapper = SpecialOrthogonalConstraint((n_factors, n_factors))
    random_params = 10 * np.random.rand(rotation_wrapper.n_params)
    rotation_wrapper.update_params(random_params)

    print(rotation_wrapper.matrix)

    rotated_factors = factors @ rotation_wrapper.matrix.T

    print(f"Matrix: {rotation_wrapper.matrix.T}")

    start = time.time()
    W1 = NonParametricEstimation().calc_rotation_matrix(rotated_factors, 4, 100)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start}")
    print(f"Estimated Matrix: {W1}")

    start = time.time()
    W2 = MattesonAndTsay().calc_rotation_matrix(rotated_factors, 4, 100)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start}")
    print(f"Estimated Matrix: {W2}")

    # plt.plot(factors)
    # plt.plot(rotated_factors @ W1.T)
    # plt.plot(rotated_factors @ W2.T)
    # plt.show()



