import numpy as np


def validate_covariance(Q):
    """Check that Q is a valid covariance matrix."""
    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Covariance matrix Q must be square")
    if not np.allclose(Q, Q.T):
        raise ValueError("Covariance matrix Q must be symmetric")
    # Check positive semi-definite
    eigenvalues = np.linalg.eigvalsh(Q)
    if np.any(eigenvalues < -1e-12):  # allow tiny numerical errors
        raise ValueError("Covariance matrix Q must be positive semi-definite")


def validate_chol_factor(LQ):
    """Check that LQ is lower triangular with positive diagonal."""
    if not np.allclose(LQ, np.tril(LQ)):
        raise ValueError("LQ must be lower triangular")
    if np.any(np.diag(LQ) < 0):
        raise ValueError("Diagonal entries of LQ must be non-negative")


def covariance_to_correlation(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a covariance matrix to a correlation matrix.

    Args:
        cov_matrix (np.ndarray): A symmetric covariance matrix (n x n).

    Returns:
        np.ndarray: The corresponding correlation matrix (n x n).
    """
    # Ensure the covariance matrix is square,
    # This is done cause its very fast and needed, no check performed for pd
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    std_devs = np.sqrt(np.diagonal(cov_matrix))
    std_matrix = np.outer(std_devs, std_devs)
    corr_matrix = cov_matrix / std_matrix
    np.fill_diagonal(corr_matrix, 1)

    return corr_matrix


class NaturalMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, int) and instance >= 0

class Natural(metaclass=NaturalMeta):
    """A type representing natural numbers (0, 1, 2, ...) for isinstance checks."""
    pass

class PositiveNaturalMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, int) and instance > 0

class PositiveNatural(metaclass=PositiveNaturalMeta):
    """A type representing positive natural numbers (1, 2, 3, ...) for isinstance checks."""
    pass

class PositiveMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, float) and instance > 0

class Positive(metaclass=PositiveMeta):
    """A type representing positive integers (1, 2, 3, ...) for isinstance checks."""
    pass
