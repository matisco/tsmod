import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space, expm
from scipy.stats import chi2

# TODO: Write this properly. Some methods are to calculate Mattesons and Tsay test stat. move it there


class NonParametricEstimation:

    @staticmethod
    def calc_rotation_matrix(factors, p, n_starting_vectors):
        n_factors = factors.shape[1]
        if n_factors == 1:
            return np.array([[1]])

        dof = n_factors * (n_factors - 1) // 2

        Ss = NonParametricEstimation.calc_Ss(factors, p) # * factors.shape[0]

        obj = lambda x: NonParametricEstimation.objectiveFunc_factorRotation(x, factors, p, Ss)

        U = np.random.rand(dof, n_starting_vectors) * 2 * np.pi - np.pi
        fvals = np.zeros((n_starting_vectors,))
        xs = np.zeros((n_starting_vectors, dof))

        options = {'maxiter': 100, 'disp': False}

        for i in range(n_starting_vectors):
            x0 = U[:, i]
            res = minimize(obj, x0, method='SLSQP', options=options) # SLSQP
            fvals[i] = res.fun
            xs[i, :] = res.x

        # Assuming x should be the one with the minimal function value
        best_index = np.argmin(fvals)
        best_x = xs[best_index, :]

        W = NonParametricEstimation.rotation_matrix(best_x, n_factors)
        return W

    @staticmethod
    def modelling_multiple_time_series_via_common_factors(data, p, nStartingVectors):
        """
        modeling of multiple time series via common factors.

        Parameters:
            data (np.ndarray): Input data matrix (T x N)
            p (int): Lag order
            nStartingVectors (int): Number of random starting vectors for optimization

        Returns:
            A (np.ndarray): Orthonormal matrix complementing B
            B (np.ndarray): Matrix of identified factor loadings
            C_inv_sqrt () : "Whitening matrix"
        """
        Y_white, C_inv_sqrt = NonParametricEstimation.whiten(data)
        N = Y_white.shape[1]

        if N == 1:
            return np.array([[1]]), np.empty((N, 0)), C_inv_sqrt

        Ss = NonParametricEstimation.calc_Ss(Y_white, p + 1)
        Ss = Ss[: ,: ,1:]  # Remove the first element

        B = np.empty((N, 0))
        p_vals = []

        while True:
            nullBT = None
            if B.size != 0:
                nullBT = null_space(B.T)

            obj = lambda x: NonParametricEstimation.objectiveFunc_identifyWN(x, Ss, p, B, nullBT)

            d = data.shape[1]
            if B.size != 0:
                d -= B.shape[1]

            if d == 1:
                break

            # Random initialization of angles
            U = np.random.rand(d - 1, nStartingVectors) * np.pi
            U[-1, :] = U[-1, :] * 2


            fvals = np.zeros((nStartingVectors,))
            xs = np.zeros((nStartingVectors, d - 1))

            options = {'maxfun': 100_000, 'maxiter': 10_000, 'disp': False}

            for i in range(nStartingVectors):
                x0 = U[:, i]
                result = minimize(obj, x0, method='L-BFGS-B', options=options)
                fvals[i] = result.fun
                xs[i, :] = result.x

            idx = np.argmin(fvals)
            best_x = xs[idx, :]
            best_x = best_x.reshape(-1)

            if B.size == 0:
                best_x = NonParametricEstimation.angles_to_unit_vector(best_x)
            else:
                best_x = NonParametricEstimation.angles_to_unit_vector(best_x)
                best_x = NonParametricEstimation.reduced_to_full_dim(best_x, nullBT)

            L = NonParametricEstimation.LBP_pm(Y_white, best_x, p, Ss, B)
            m = B.shape[1] + 1
            p_val = 1 - chi2.cdf(L, p * (2 * m - 1))

            p_vals.append(p_val)
            if p_val < 0.01:
                break
            B = np.column_stack((B, best_x))


        A = null_space(B.T)

        return A, B, C_inv_sqrt

    @staticmethod
    def existence_test_stat(h, m):
        T, n = h.shape

        def corrcoef_(h, i, j, l):
            # Equivalent to the custom covariance logic you implemented in MATLAB
            # return np.sum(h[l:, i] * h[:T - l, j]) / (T - l - 1)
            return np.corrcoef(h[l:, i], h[:T - l, j])[0 ,1]

        sum1 = 0.0
        for j in range(n):
            for i in range(j):
                sum1 += corrcoef_(h, i, j, 0) ** 2

        sum2 = 0.0
        for k in range(1, m + 1):
            inner_sum = 0.0
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    inner_sum += corrcoef_(h, i, j, k) ** 2 / (T - k)
            sum2 += inner_sum

        Q = T * sum1 + T * (T + 2) * sum2
        p_val = 1 - chi2.cdf(Q, n * (n - 1) / 2 + m * n * (n - 1))

        return Q, p_val

    @staticmethod
    def whiten(Y):
        # Step 1: Center the data
        Y_mean = np.mean(Y, axis=0)  # Mean across columns (samples)
        Y_centered = Y - Y_mean

        # Step 2: Compute the covariance matrix
        C = np.cov(Y_centered, rowvar=False, bias=True)

        if Y.shape[1] > 1:

            # Step 3: Inverse square root of the covariance matrix
            D, V = np.linalg.eigh(C)  # Use eigh since C is symmetric
            D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
            C_inv_sqrt = V @ D_inv_sqrt @ V.T

            # Step 4: Apply the whitening transform
            Y_white = Y_centered @ C_inv_sqrt.T
        else:
            C_inv_sqrt = 1/ np.sqrt(C)
            Y_white = Y_centered * C_inv_sqrt

        return Y_white, C_inv_sqrt

    @staticmethod
    def phi_b(Ss, p, b):
        b = b.reshape(-1, 1)
        phi = 0
        for k in range(p):
            phi += (b.T @ Ss[:, :, k] @ b) ** 2

        return float(phi)

    @staticmethod
    def phi_m_b(Ss, p, b, B):
        b = b.reshape(-1, 1)
        m = B.shape[1] + 1

        phi = 0
        for k in range(p):
            for i in range(m - 1):
                phi += (b.T @ Ss[:, :, k] @ B[:, i]) ** 2 + (B[:, i].T @ Ss[:, :, k] @ b) ** 2

        return float(phi)

    @staticmethod
    def LBP_p1(Y, b, p, Ss):
        b = b.reshape(-1, 1)
        T = Y.shape[0]

        L = 0
        for k in range(p):
            L += ((b.T @ Ss[:, :, k] @ b) ** 2) / (T - k)

        return (T * (T + 2) * L)

    @staticmethod
    def LBP_pm(Y, b, p, Ss, B):

        if B.size == 0:
            return NonParametricEstimation.LBP_p1(Y, b, p, Ss)

        b = b.reshape(-1, 1)
        T = Y.shape[0]
        m = B.shape[1] + 1

        L = 0
        for k in range(p):
            inner_sum = 0
            for j in range(m - 1):
                inner_sum += (b.T @ Ss[:, :, k] @ B[:, j]) ** 2 + (B[:, j].T @ Ss[:, :, k] @ b) ** 2

            L += (1 / (T - k)) * ((b.T @ Ss[:, :, k] @ b) ** 2 + inner_sum)

        return (T * T * L)

    @staticmethod
    def calc_Ss2(x, m):
        dim = x.shape[1]
        Ss = np.zeros((dim, dim, m + 1))

        for j in range(dim):
            for i in range(dim):
                Ss[i, j, 0] = np.sum(x[:, i] * x[:, j])

        for l in range(1, m + 1):
            for j in range(dim):
                for i in range(dim):
                    Ss[i, j, l] = np.sum(x[l:, i] * x[:-l, j])

        return Ss

    @staticmethod
    def calc_Ss(x: np.ndarray, m: int) -> np.ndarray:
        """
        Compute lagged inner product matrices for a range of lags up to m.

        Parameters:
        x (np.ndarray): Input array of shape (n_samples, dim).
        m (int): Maximum lag.

        Returns:
        np.ndarray: 3D array of shape (dim, dim, m+1) containing sums of outer products at each lag.
        """
        T, dim = x.shape
        Ss = np.zeros((dim, dim, m + 1))

        for l in range(m + 1):
            if l == 0:
                product = x.T @ x
            else:
                product = x[l:].T @ x[:-l]
            Ss[:, :, l] = product / T

        return Ss

    @staticmethod
    def skew_symmetric_from_params(theta, n):
        A = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                A[i, j] = theta[idx]
                A[j, i] = -theta[idx]
                idx += 1

        return A

    @staticmethod
    def rotation_matrix(theta, n):
        A = NonParametricEstimation.skew_symmetric_from_params(theta, n)
        return expm(A)

    @staticmethod
    def objectiveFunc_factorRotation(theta, s, m, Ss):
        n = s.shape[1]
        T = s.shape[0]

        A = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                A[i, j] = theta[idx]
                A[j, i] = -theta[idx]
                idx += 1

        W = expm(A)

        obj = 0

        valid_idx = np.ones((n, n), dtype=bool) & (~np.eye(n, dtype=bool))

        # l = 0 case
        l = 0
        current_weight = 2 * (1 - l / (m + 1))
        current_S = W @ Ss[:, :, l] @ W.T
        obj += current_weight * np.sum(np.tril(current_S)[valid_idx.T] ** 2)  # tril picks lower triangle

        for l in range(1, m + 1):
            current_weight = (1 - l / (m + 1))
            current_S = W @ Ss[:, :, l] @ W.T
            obj += current_weight * np.sum(current_S[valid_idx] ** 2)

        return obj

    @staticmethod
    def angles_to_unit_vector(theta):
        """
        Convert hyperspherical angles to a unit vector in R^d.

        Parameters:
            theta (array-like): (d-1,) vector of angles
                                theta[0] to theta[d-2] ∈ [0, pi]
                                theta[d-1] ∈ [0, 2*pi]

        Returns:
            x (np.ndarray): d-dimensional unit vector
        """
        theta = np.asarray(theta)
        d = len(theta) + 1
        x = np.zeros(d)
        prod_sin = 1.0

        for i in range(d):
            if i == 0:
                x[i] = np.cos(theta[0])
            elif i < d - 1:
                prod_sin *= np.sin(theta[i - 1])
                x[i] = prod_sin * np.cos(theta[i])
            else:
                prod_sin *= np.sin(theta[d - 2])
                x[i] = prod_sin

        return x

    @staticmethod
    def reduced_to_full_dim(x1, N):
        """
        Convert a reduced-dimension vector x1 back to full dimension using basis N.

        Parameters:
            x1 (array-like): vector in reduced dimension (length ortho_dim)
            N (np.ndarray): full_dim x ortho_dim matrix (each column is a basis vector)

        Returns:
            x (np.ndarray): full_dim vector
        """
        x1 = np.asarray(x1)
        N = np.asarray(N)

        full_dim, ortho_dim = N.shape
        x = np.zeros(full_dim)

        for j in range(ortho_dim):
            x += x1[j] * N[:, j]

        return x

    @staticmethod
    def objectiveFunc_identifyWN(x, Ss, p, B, nullBT):
        """
        Unconstrained objective function.

        Parameters:
            x      : input vector (angles)
            Ss     : data or parameter needed for phi functions
            p      : parameter needed for phi functions
            B      : matrix or None/empty
            nullBT : basis matrix for reduced to full dimension conversion

        Returns:
            res    : negative value of the objective function
        """
        if B.size == 0:
            x_vec = NonParametricEstimation.angles_to_unit_vector(x)
            return NonParametricEstimation.phi_b(Ss, p, x_vec)
        else:
            x_vec = NonParametricEstimation.angles_to_unit_vector(x)
            x_full = NonParametricEstimation.reduced_to_full_dim(x_vec, nullBT)
            return (NonParametricEstimation.phi_b(Ss, p, x_full) + NonParametricEstimation.phi_m_b(Ss, p, x_full, B))

