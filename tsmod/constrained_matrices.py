# import warnings
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
# from functools import cached_property
from typing import Literal
from scipy.linalg import expm, logm
import inspect

from base import Signal, check_is_defined, CompositeSignal

from utils import (softplus,
                   softplus_inv,
                   sigmoid_softplus_inv,
                   numerical_jacobian,
                   indexing_column_to_row_major,
                   indexing_row_to_column_major)


# TODO:
#  1. there are some inconsistencies with how _n_params is used and specifically how it interacts with has_shape
#  2. currently whenever parameters are updated calculations which may be expensive are performed.
#           It could be done lazily, doubt it makes a difference.
#           If parameters are updated, one would think the matrix will be called

# -----------
# Abstract classes. Base, composite and with underlying
#----------

class ConstrainedMatrix(Signal, ABC):
    """
    Abstract base class for matrices with constraints and parameterized representations.

    This class provides a framework for representing a matrix in terms of
    unconstrained parameters while enforcing constraints on rows, columns, or the
    entire matrix. It is designed to support optimization, EM algorithms, and
    constrained solvers.

    Key responsibilities:
    --------------------
    - Parameterization:
        - get_params(): Return the unconstrained free parameters of the matrix.
        - update_params(params): Update the matrix from a given set of free parameters.

    - Constraint handling:
        - get_row_constraint(idx): Return linear equality and inequality constraints
          for a specific row. Returns a tuple (A_eq, b_eq, A_ineq, b_ineq), where
          A_eq @ row = b_eq and A_ineq @ row >= b_ineq.
        - get_col_constraint(idx): Same as above, but for columns.

    - Jacobian computation:
        - jacobian_vec_params(order='C'|'F'): Return the Jacobian of the vectorized
          matrix (row-major 'C' or column-major 'F') with respect to the parameters.
          The Jacobian has shape (n*m, n_params) for a matrix of shape (n, m).

    Notes:
    ------
    - Subclasses may implement specialized methods for Jacobians:
        - _jacobian_vecC_params() for column-major
        - _jacobian_vecF_params() for row-major
      If these exist, they are used; otherwise, a numerical approximation can be used.

    - Useful for:
        - Gradient-based optimization with unconstrained parameters.
        - Constrained optimization with linear equality/inequality constraints.
        - EM algorithms requiring derivatives of vectorized matrices.

    """

    _enforce_square: bool | None = None
    _is_elementwise: bool | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if inspect.isabstract(cls):
            return

        # enforce_square must be class-level
        if cls._enforce_square is None:
            raise TypeError(
                f"{cls.__name__} must define class variable '_enforce_square'"
            )

        # is_elementwise can be class-level OR property
        if (
            cls._is_elementwise is None
            and not isinstance(getattr(cls, "_is_elementwise", None), property)
        ):
            raise TypeError(
                f"{cls.__name__} must define '_is_elementwise' "
                f"as a class attribute or property"
            )

    def __init__(self,
                 shape: tuple):

        super().__init__(shape)
        if self._enforce_square:
            if (shape[0] is not None) or (shape[1] is not None):
                if shape[0] != shape[1]:
                    raise ValueError("Shape of matrix must be square")

        self._matrix = None

    # def __getitem__(self, key):
    #     return self._matrix[key]

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix_setter(matrix, run_check=True)

    def set_matrix(self, value):
        self._matrix_setter(value, run_check=True)
        return self

    def set_matrix_trusted(self, value):
        self._matrix_setter(value, run_check=False)
        return self

    def _matrix_setter(self, value, run_check: bool):
        if not self.has_shape:
            shape = value.shape
            if self._enforce_square and (shape[0] != shape[1]):
                raise ValueError("Shape of matrix must be square")
            self.shape = shape

        if value.shape != self.shape:
            raise ValueError(f"Matrix shape mismatch: {value.shape} != {self.shape}")
        if run_check:
            self._check_matrix_constrain(value)
        self._on_matrix_setter(value)
        self._matrix = value

    def _on_matrix_setter(self, matrix):
        """Hook for subclasses to override if needed."""
        pass

    @property
    def is_defined(self):
        return self._matrix is not None

    @abstractmethod
    def _check_matrix_constrain(self, matrix):
        raise NotImplementedError

    @abstractmethod
    def _get_params(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _update_params(self, params) -> None:
        raise NotImplementedError

    def get_row_constraint(self, idx: int):
        if not self._is_elementwise:
            raise NotImplementedError("No row wise constraints for 'non-element-wise' constrained matrices")

        if not isinstance(idx, int):
            raise ValueError(f"idx must be int")

        if idx < 0 or idx >= self.shape[0]:
            raise ValueError(f"idx must be between 0 and {self.shape[0] - 1}")

        return self._get_row_constraint(idx)

    def _get_row_constraint(self, idx: int):
        raise NotImplementedError

    def get_col_constraint(self, idx: int):
        if not self._is_elementwise:
            raise NotImplementedError("No col wise constraints for 'non-element-wise' constrained matrices")

        if not isinstance(idx, int):
            raise ValueError(f"idx must be int")

        if idx < 0 or idx >= self.shape[1]:
            raise ValueError(f"idx must be between 0 and {self.shape[1] - 1}")

        # raise NotImplementedError
        return self._get_col_constraint(idx)

    def _get_col_constraint(self, idx: int):
        raise NotImplementedError

    @check_is_defined
    def get_vectorized(self, order: Literal['F', 'C'] = 'C'):
        return self.matrix.flatten(order=order)

    @check_is_defined  # not strickly necessary but makes my life easier
    def jacobian_vec_params(self, order: Literal['F', 'C'] = 'C'):
        order_map = {
            'F': ('_jacobian_vecF_params', '_jacobian_vecC_params', indexing_row_to_column_major),
            'C': ('_jacobian_vecC_params', '_jacobian_vecF_params', indexing_column_to_row_major)
        }

        if order not in order_map:
            raise ValueError("order must be 'F' or 'C'")

        primary_attr, fallback_attr, index_func = order_map[order]

        if hasattr(self, primary_attr):
            return getattr(self, primary_attr)()
        elif hasattr(self, fallback_attr):
            j = getattr(self, fallback_attr)()
            arr = index_func(*self.shape)
            return j[arr]
        else:
            return self.numerical_jacobian_vec_params(order)

    @check_is_defined
    def numerical_jacobian_vec_params(self, order: Literal['F', 'C'] = 'C'):
        params_now = self.get_params()
        def func(x):
            self._update_params(x)
            return self.get_vectorized(order)
        Jac = numerical_jacobian(func, params_now)
        self._update_params(params_now)
        return Jac


class CompositeConstraintMatrix(CompositeSignal, ConstrainedMatrix):

    _enforce_square = False

    def __init__(self, matrices: list[ConstrainedMatrix], combine: Literal['vstack', 'hstack', 'blockdiag']):
        """
        Parameters
        ----------
        matrices : list[ConstrainedMatrix]
            The matrices to combine.
        combine : {'vstack', 'hstack', 'blockdiag'}
            How to combine the matrices:
            - 'vstack': vertically stack the matrices
            - 'hstack': horizontally stack the matrices
            - 'blockdiag': place matrices along the block diagonal
        """
        self._underlying_matrices = matrices
        self.combine = combine

        shape = self._check_init_and_infer_shape()

        super().__init__(shape)

    @property
    def _is_elementwise(self) -> bool:
        return all(m._is_elementwise for m in self._underlying_matrices)

    def _check_init_and_infer_shape(self):
        """
        Validate and propagate shape constraints for composite constrained matrices.

        Rules
        -----
        vstack:
            - Number of rows must be known for every matrix
            - All matrices must share the same number of columns (if specified)
        hstack:
            - Number of columns must be known for every matrix
            - All matrices must share the same number of rows (if specified)
        blockdiag:
            - Shapes must be fully specified
        """
        if not self._underlying_matrices:
            raise ValueError("No matrices provided")

        if self.combine == "vstack":
            # Rows must be known to define total size
            tot_shape0 = 0
            for i, m in enumerate(self._underlying_matrices):
                if m.shape[0] is None:
                    raise ValueError(
                        f"Matrix {i} has unknown number of rows; "
                        "rows must be specified for vstack"
                    )

                tot_shape0 += m.shape[0]

            # Infer / enforce common columns
            ref_cols = next(
                (m.shape[1] for m in self._underlying_matrices if m.shape[1] is not None),
                None,
            )

            if ref_cols is not None:
                for m in self._underlying_matrices:
                    m.shape = (m.shape[0], ref_cols)

            return tot_shape0, ref_cols

        elif self.combine == "hstack":
            # Columns must be known to define total size
            tot_shape1 = 0
            for i, m in enumerate(self._underlying_matrices):
                if m.shape[1] is None:
                    raise ValueError(
                        f"Matrix {i} has unknown number of columns; "
                        "columns must be specified for hstack"
                    )
                tot_shape1 += m.shape[1]

            # Infer / enforce common rows
            ref_rows = next(
                (m.shape[0] for m in self._underlying_matrices if m.shape[0] is not None),
                None,
            )

            if ref_rows is not None:
                for m in self._underlying_matrices:
                    m.shape = (ref_rows, m.shape[1])

            return ref_rows, tot_shape1

        elif self.combine == "blockdiag":
            tot_shape0 = 0
            tot_shape1 = 0
            for i, m in enumerate(self._underlying_matrices):
                if None in m.shape:
                    raise ValueError(
                        f"Matrix {i} has shape {m.shape}; "
                        "blockdiag requires fully specified shapes"
                    )
                tot_shape0 += m.shape[0]
                tot_shape1 += m.shape[1]

            return tot_shape0, tot_shape1

        else:
            raise ValueError(f"Unknown combine mode: {self.combine}")

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int | None, int | None]):
        if not isinstance(value, tuple):
            raise TypeError(f"shape must be a tuple, got {type(value)}")
        if len(value) != 2:
            raise ValueError(f"shape must be length 2, got length {len(value)}")
        if not all(isinstance(x, (int, type(None))) for x in value):
            raise TypeError("Each element of shape must be an int or None")

        shape = self.check_or_infer_shape(value)

        if self.combine == "vstack":
            for m in self._underlying_matrices:
                if m is not self:
                    m.shape = (None, shape[1])
        elif self.combine == "hstack":
            for m in self._underlying_matrices:
                if m is not self:
                    m.shape = (shape[0], None)

        self._shape = shape

    def _iterate_blocks(self, matrix):
        """
        Yield (submatrix, constrained_matrix) pairs according to the
        composite structure.
        """
        if self.combine == "vstack":
            idx = 0
            for m in self._underlying_matrices:
                rows = m.shape[0]
                yield matrix[idx:idx + rows, :], m
                idx += rows

        elif self.combine == "hstack":
            idx = 0
            for m in self._underlying_matrices:
                cols = m.shape[1]
                yield matrix[:, idx:idx + cols], m
                idx += cols

        elif self.combine == "blockdiag":
            idx0 = idx1 = 0
            for m in self._underlying_matrices:
                r, c = m.shape
                yield matrix[idx0:idx0 + r, idx1:idx1 + c], m
                idx0 += r
                idx1 += c

        else:
            raise ValueError(f"Unknown combine mode: {self.combine}")

    def _on_matrix_setter(self, matrix):
        for submat, m in self._iterate_blocks(matrix):
            m.matrix = submat

    @property
    def _underlying_signals(self) -> list[Signal]:
        return self._underlying_matrices

    def _check_matrix_constrain(self, matrix):
        for submat, m in self._iterate_blocks(matrix):
            m._check_matrix_constrain(submat)

    def _get_col_constraint(self, idx: int):
        raise NotImplementedError

    def _get_row_constraint(self, idx: int):
        raise NotImplementedError


class ConstrainedMatrixWithUnderlying(ConstrainedMatrix, ABC):

    def __init__(self,
                 shape: tuple,
                 underlying: ConstrainedMatrix,
                 store_both: bool,
                 run_check_on: Literal['top','underlying', 'both']):

        super().__init__(shape)
        self._underlying = underlying
        self._store_both = store_both
        self._run_check_on = run_check_on

    @abstractmethod
    def _on_shape_setter(self, shape: tuple):
        raise NotImplementedError

    @property
    def matrix(self):
        if self._store_both and self._matrix is not None:
            return self._matrix

        matrix = self._compute_matrix_from_underlying()

        if self._store_both:
            self._matrix = matrix

        return matrix

    @property
    def is_defined(self):
        return self._underlying.is_defined

    def _matrix_setter(self, value, run_check: bool):
        # 1. Set shape if unknown
        if not self.has_shape:
            shape = value.shape
            if self._enforce_square and (shape[0] != shape[1]):
                raise ValueError("Shape of matrix must be square")
            self.shape = shape

        # 2. Validate shape
        if value.shape != self.shape:
            raise ValueError(f"Matrix shape mismatch: {value.shape} != {self.shape}")

        # 3. Check constraints on this matrix if needed
        if run_check and (self._run_check_on in ('both', 'top')):
            self._check_matrix_constrain(value)

        # 4. Hook for subclasses
        self._on_matrix_setter(value)

        # 5. Optionally store the matrix
        if self._store_both:
            self._matrix = value

        # 6. Compute underlying
        underlying_matrix = self._compute_underlying_from_matrix(value)

        # 7. Update underlying with or without checks
        if run_check and (self._run_check_on in ('both', 'underlying')):
            self._underlying.set_matrix(underlying_matrix)
        else:
            self._underlying.set_matrix_trusted(underlying_matrix)

    @abstractmethod
    def _compute_matrix_from_underlying(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _compute_underlying_from_matrix(self, matrix):
        raise NotImplementedError

    @abstractmethod
    def _check_matrix_constrain(self, matrix):
        raise NotImplementedError

    @property
    def _n_params(self) -> int:
        return self._underlying.n_params

    def _update_params(self, params) -> None:
        self._underlying.update_params(params)
        self._matrix = None

    def _get_params(self) -> np.ndarray:
        return self._underlying.get_params()

# -----------
# Concrete implementations
#----------


# -----------
# Unconstrained
#----------

class FreeMatrix(ConstrainedMatrix):
    _enforce_square = False
    _is_elementwise = True

    def _check_matrix_constrain(self, matrix):
        pass

    def _get_params(self) -> np.ndarray:
        return self.matrix.flatten(order='F')

    def _update_params(self, params) -> None:
        self.matrix = params.reshape(self.shape, order='F')

    def _get_row_constraint(self, idx: int):
        return None, None, None, None

    def _get_col_constraint(self, idx: int):
        return None, None, None, None

    def _jacobian_vecF_params(self):
        return np.eye(self.shape[0] * self.shape[1])

    @property
    def _n_params(self):
        if self.has_shape:
            return self.shape[0] * self.shape[1]
        else:
            raise ValueError("Matrix has no defined shape. Unknown number of parameters")

# -----------
# Fully Constrained
#----------

class FullyDefinedMatrix(ConstrainedMatrix, ABC):
    _is_elementwise = True

    def __init__(self, shape: tuple[int, int]):
        super().__init__(shape)
        self._set_matrix()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        ConstrainedMatrix.shape.fset(self, value)
        self._set_matrix()

    @abstractmethod
    def _set_matrix(self):
        raise NotImplementedError

    @property
    def _n_params(self):
        return 0

    def _get_params(self) -> np.ndarray:
        return np.array([])

    def _update_params(self, params) -> None:
        if len(params):
            raise ValueError("matrix is fully defined, params must be len 0")

    def _jacobian_vecF_params(self):
        return np.array([])

    def _jacobian_vecC_params(self):
        return np.array([])


class ZeroMatrix(FullyDefinedMatrix):
    _enforce_square = False

    def _check_matrix_constrain(self, matrix):
        if not np.allclose(matrix, 0):
            raise ValueError("Matrix must be zero if constrain is zero")

    def _get_row_constraint(self, idx: int):
        n, m = self.shape
        A = np.eye(m)
        b = np.zeros((m,))
        return A, b.reshape(-1, 1), None, None

    def _get_col_constraint(self, idx: int):
        n, m = self.shape
        A = np.eye(n)
        b = np.zeros((n,))
        return A, b.reshape(-1, 1), None, None

    def _set_matrix(self):
        if self.has_shape:
            self.matrix = np.zeros(self.shape)


class IdentityMatrix(FullyDefinedMatrix):
    _enforce_square = False


    def _check_matrix_constrain(self, matrix):
        matrix_1 = np.zeros(self.shape)
        idx_diag = np.diag_indices(min(self.shape))
        matrix_1[idx_diag] = 1
        diff = matrix - matrix_1
        if not np.allclose(diff, 0):
            raise ValueError("Matrix must be identity if constrain is identity")

    def _get_row_constraint(self, idx: int):
        n, m = self.shape
        A = np.eye(m)
        b = np.zeros((m,))
        if idx >= m:
            return A, b.reshape(-1, 1), None, None
        else:
            b[idx] = 1  # Set the idx-th value to 1
            return A, b.reshape(-1, 1), None, None

    def _get_col_constraint(self, idx: int):
        n, m = self.shape
        A = np.eye(n)
        b = np.zeros((n,))
        if idx >= n:
            return A, b.reshape(-1, 1), None, None
        else:
            b[idx] = 1  # Set the idx-th value to 1
            return A, b.reshape(-1, 1), None, None

    def _set_matrix(self):
        if self.has_shape:
            matrix = np.zeros(self.shape)
            matrix[np.diag_indices(min(self.shape))] = 1
            self.matrix = matrix


class FullyDefinedConstructor(FullyDefinedMatrix):
    _enforce_square = False

    def __init__(self, matrix: np.ndarray):
        self._matrix = matrix
        super().__init__(matrix.shape)

    def _set_matrix(self):
        pass

    def _get_col_constraint(self, idx: int):
        n, m = self.shape
        A = np.eye(n)
        b = self._matrix[:, idx]
        return A, b.reshape(-1,1), None, None

    def _get_row_constraint(self, idx: int):
        n, m = self.shape
        A = np.eye(m)
        b = self._matrix[idx+idx+1, :]
        return A, b, None, None

    def _check_matrix_constrain(self, matrix):
        if not np.allclose(matrix, self._matrix):
            raise ValueError("Matrix is fully defined")

# -----------
# Misc
#----------

class UnitTopRowConstrained(ConstrainedMatrix):
    _is_elementwise = True
    _enforce_square = False

    def _check_matrix_constrain(self, matrix):
        if not np.allclose(matrix[0, :], 1):
            raise ValueError("Top row must be ones")

    def _get_row_constraint(self, idx: int):
        if idx == 0:
            n, m = self.shape
            A = np.eye(m)
            b = np.ones((m, 1))
            return A, b, None, None

        return None, None, None, None

    def _update_params(self, params) -> None:
        if self.matrix is None:
            matrix = np.zeros(self.shape)
            matrix[0, :] = 1
            self.matrix = matrix

        n, m = self.shape
        self.matrix[1:, :] = params.reshape((n-1, m), order='F')

    def _get_params(self) -> np.ndarray:
        # return self._matrix[1:, :].copy().reshape(-1, order='F')
        slice_2d = self._matrix[1:, :]
        return slice_2d.ravel(order='F')

    def _get_col_constraint(self, idx: int):
        n, m = self.shape
        A = np.zeros((1, n))
        A[0, 0] = 1
        b = np.ones((1, 1))
        return A, b, None, None

    @property
    def _n_params(self) -> int:
        n, m = self.shape
        return (n - 1) * m

    def _jacobian_vecF_params(self):
        size = self.shape[0] * self.shape[1]
        j = np.eye(size)
        arr = np.ones((size,), dtype=bool)
        arr[np.arange(0, size, self.shape[0])] = False
        return j[:, arr]

# -----------
# Diagonal
#----------


class DiagonalMatrix(ConstrainedMatrix):
    _is_elementwise = True
    _enforce_square = False

    def _check_matrix_constrain(self, matrix):
        n, m = self.shape
        idx_all = np.ones((n, m), dtype=bool)
        idx_all[np.diag_indices(min(n, m))] = False
        if not np.allclose(matrix[idx_all], 0):
            raise ValueError("Matrix must be diagonal if constrain is diagonal")

    def _get_params(self) -> np.ndarray:
        return np.diagonal(self._matrix)

    def _update_params(self, params) -> None:
        if self.matrix is None:
            self.matrix = np.zeros(self.shape)
        self._matrix[np.diag_indices(min(self.shape))] = params

    def _get_row_constraint(self, idx: int):
        n, m = self.shape
        A = np.eye(m)
        if idx >= m:
            b = np.zeros((m,))
            return A, b, None, None
        else:
            A = np.delete(A, idx, axis=0)  # Remove the row
            b = np.zeros((m - 1,))
            return A, b.reshape(-1, 1), None, None

    def _get_col_constraint(self, idx: int):
        n, m = self.shape
        A = np.eye(n)
        if idx >= n:
            b = np.zeros((n,))
            return A, b, None, None
        else:
            A = np.delete(A, idx, axis=0)  # Remove the row
            b = np.zeros((n - 1,))
            return A, b.reshape(-1, 1), None, None

    def _jacobian_vecF_params(self):
        j = np.zeros((self.shape[0] * self.shape[1], self.n_params))
        for i in range(self.n_params):
            j[i*self.shape[0]+i, i] = 1
        return j

    def _jacobian_vecC_params(self):
        j = np.zeros((self.shape[0] * self.shape[1], self.n_params))
        for i in range(self.n_params):
            j[i*self.shape[1]+i, i] = 1
        return j

    @property
    def _n_params(self):
        if self.has_shape:
            return min(*self.shape)
        else:
            raise ValueError("Matrix has no defined shape. Unknown number of parameters")


class PosDiagonalMatrix(ConstrainedMatrix):
    _is_elementwise = True
    _enforce_square = False

    def _check_matrix_constrain(self, matrix):
        n, m = self.shape
        idx_all = np.ones((n, m), dtype=bool)
        idx_diag = np.diag_indices(min(n, m))
        idx_all[idx_diag] = False
        if not np.allclose(matrix[idx_all], 0):
            raise ValueError("Matrix must be diagonal if constrain is diagonal")

        if not np.all(matrix[idx_diag] > 0):
            raise ValueError("Matrix must have positive diagonal elements if constrain is pos_diagonal")

    def _get_params(self) -> np.ndarray:
        return softplus_inv(np.diagonal(self.matrix))

    def _update_params(self, params) -> None:
        if self.matrix is None:
            matrix = np.zeros(self.shape)
            matrix[np.diag_indices(min(self.shape))] = softplus(params)
            self.matrix = matrix
        else:
            self._matrix[np.diag_indices(min(self._shape))] = softplus(params)

    def _get_row_constraint(self, idx: int):
        n, m = self.shape
        A = np.eye(m)
        if idx >= m:
            b = np.zeros((m,))
            return A, b.reshape(-1, 1), None, None
        else:
            A = np.delete(A, idx, axis=0)
            b = np.zeros((m - 1,))

            # Setup inequality constraint: x[idx] > 0
            A_ineq = np.zeros((1, m))
            A_ineq[0, idx] = 1  # Positive constraint on the idx-th element
            b_ineq = np.array([0])
            return A, b.reshape(-1, 1), A_ineq, b_ineq.reshape(-1, 1)

    def _get_col_constraint(self, idx: int):
        n, m = self.shape
        A = np.eye(n)
        if idx >= n:
            b = np.zeros((n,))
            return A, b, None, None
        else:
            A = np.delete(A, idx, axis=0)  # Remove the row
            b = np.zeros((n - 1,))

            # Setup inequality constraint: x[idx] > 0
            A_ineq = np.zeros((1, n))
            A_ineq[0, idx] = 1  # Positive constraint on the idx-th element
            b_ineq = np.array([0])

            return A, b.reshape(-1, 1), A_ineq, b_ineq.reshape(-1, 1)

    def _jacobian_vecF_params(self):
        derivatives = sigmoid_softplus_inv(np.diagonal(self.matrix))
        j = np.zeros((self.shape[0] * self.shape[1], self.n_params))
        for i in range(self.n_params):
            j[i * self.shape[0] + i, i] = derivatives[i]
        return j

    def _jacobian_vecC_params(self):
        derivatives = sigmoid_softplus_inv(np.diagonal(self.matrix))
        j = np.zeros((self.shape[0] * self.shape[1], self.n_params))
        for i in range(self.n_params):
            j[i*self.shape[1]+i, i] = derivatives[i]
        return j

    @property
    def _n_params(self):
        if self.has_shape:
            return min(*self.shape)
        else:
            raise ValueError("Matrix has no defined shape. Unknown number of parameters")

# -----------
# Lower Triangular
#----------

class LowerTriangularMatrix(ConstrainedMatrix):
    _is_elementwise = True
    _enforce_square = False

    def _check_matrix_constrain(self, matrix):
        n, m = self.shape
        idx = np.triu_indices(n=n, k=1, m=m)
        if not np.allclose(matrix[idx], 0):
            raise ValueError("Matrix must be lower triangular if constrain is lower_triangular")

    def _get_params(self) -> np.ndarray:
        n, m = self.shape
        idx = np.tril_indices(n, k=0, m=m)
        return self.matrix[idx]

    def _update_params(self, params) -> None:
        if self.matrix is None:
            self.matrix = np.zeros(self.shape)
        n, m = self.shape
        # idx = np.tril_indices(n, k=0, m=m)
        # self._matrix[idx] = params

        idx_shift = 0
        for i in range(n):
            _range = range(0, min(i + 1, m))
            self._matrix[i, _range] = params[idx_shift:idx_shift+len(_range)]
            idx_shift += len(_range)

    def _get_row_constraint(self, idx: int):
        n, m = self.shape

        zero_idxs = np.arange(idx + 1, m)
        if not len(zero_idxs):
            return None, None, None, None

        A = np.zeros((len(zero_idxs), m))
        for i, idx in enumerate(zero_idxs):
            A[i, idx] = 1
        b = np.zeros(len(zero_idxs))

        return A, b.reshape(-1, 1), None, None

    def _get_col_constraint(self, idx: int):
        n, m = self.shape

        zero_idxs = np.arange(0, idx)

        A = np.zeros((len(zero_idxs), n))
        for i, idx_ in enumerate(zero_idxs):
            A[i, idx_] = 1
        b = np.zeros(len(zero_idxs))

        return A, b.reshape(-1, 1), None, None

    def _jacobian_vecF_params(self):
        # n, m = self.shape
        # I = np.eye(n*m)
        # arr = np.ones((n*m,), dtype=bool)
        # for i in range(1, m):
        #     for j in range(i):
        #         arr[i*n+j] = False
        # return I[:, arr]

        n, m = self.shape
        arr = np.arange(n * m).reshape((n, m), order="F")

        j = np.zeros((n * m, self.n_params))
        idx = arr[np.tril_indices(n, k=0, m=m)]
        j[idx, np.arange(idx.size)] = 1

        return j

    @property
    def _n_params(self):
        if self.has_shape:
            n, m = self.shape
            if m < n:
                return m * n - (m - 1) * m // 2
            else:
                return n * (n + 1) // 2
        else:
            raise ValueError("Matrix has no defined shape. Unknown number of parameters")


class SetDiagLowerTriangularConstraint(ConstrainedMatrix):
    _is_elementwise = True
    _enforce_square = False

    def __init__(self, shape, diag_value: float):
        super().__init__(shape)
        self._diag_value = diag_value

    def _check_matrix_constrain(self, matrix):
        n, m = self.shape
        idx = np.triu_indices(n=n, k=1, m=m)
        if not np.allclose(matrix[idx], 0):
            raise ValueError("Matrix must be lower triangular if constrain is lower_triangular")

        idx_diag = np.diag_indices(min(n, m))
        if not np.allclose(matrix[idx_diag], self._diag_value):
            raise ValueError("Matrix must have unit diagonal if constrain is unit_diag_lower_triangular")

    def _get_params(self) -> np.ndarray:
        n, m = self.shape
        idx = np.tril_indices(n, k=-1, m=m)
        return self.matrix[idx]

    def _update_params(self, params) -> None:
        if self.matrix is None:
            matrix = np.zeros(self.shape)
            matrix[np.diag_indices(min(*self.shape))] = self._diag_value
            self.matrix = matrix

        n, m = self.shape
        idx = np.tril_indices(n, k=-1, m=m)
        self._matrix[idx] = params

    def _get_row_constraint(self, idx: int):
        n, m = self.shape

        zero_idxs = np.arange(idx + 1, m)

        if idx > m - 1:
            return None, None, None, None

        A = np.zeros((len(zero_idxs) + (idx < m), m))
        for i, idx_ in enumerate(zero_idxs):
            A[i, idx_] = 1
        b = np.zeros(len(zero_idxs) + (idx < m))

        if idx < m:
            A[len(zero_idxs), idx] = 1
            b[len(zero_idxs)] = self._diag_value

        return A, b.reshape(-1, 1), None, None

    def _get_col_constraint(self, idx: int):
        n, m = self.shape

        zero_idxs = np.arange(0, idx)

        A = np.zeros((len(zero_idxs) + (idx < n), n))
        for i, idx_ in enumerate(zero_idxs):
            A[i, idx_] = 1
        b = np.zeros(len(zero_idxs) + (idx < n))

        if idx < n:
            A[len(zero_idxs), idx] = 1
            b[len(zero_idxs)] = self._diag_value

        return A, b.reshape(-1, 1), None, None

    def _jacobian_vecF_params(self):
        # n, m = self.shape
        # I = np.eye(n*m)
        # arr = np.ones((n*m,), dtype=bool)
        # for i in range(0, m):
        #     for j in range(i+1):
        #         arr[i*n+j] = False
        # return I[:, arr]

        n, m = self.shape
        arr = np.arange(n * m).reshape((n, m), order="F")

        j = np.zeros((n * m, self.n_params))
        idx = arr[np.tril_indices(n, k=-1, m=m)]
        j[idx, np.arange(idx.size)] = 1

        return j

    @property
    def _n_params(self):
        if self.has_shape:
            n, m = self.shape
            if m < n:
                return m * n - (m + 1) * m // 2
            else:
                return n * (n - 1) // 2
        else:
            raise ValueError("Matrix has no defined shape. Unknown number of parameters")


class UnitLowerTriMatrix(SetDiagLowerTriangularConstraint):

    def __init__(self, shape):
        super().__init__(shape, 1)


class ZeroLowerTriMatrix(SetDiagLowerTriangularConstraint):

    def __init__(self, shape):
        super().__init__(shape, 0)


class PosDiagLowerTriMatrix(ConstrainedMatrix):
    _is_elementwise = True
    _enforce_square = False

    def _check_matrix_constrain(self, matrix):
        n, m = self.shape
        idx = np.triu_indices(n=n, k=1, m=m)
        if not np.allclose(matrix[idx], 0):
            raise ValueError("Matrix must be lower triangular if constrain is lower_triangular")

        idx_diag = np.diag_indices(min(n, m))
        if not np.all(matrix[idx_diag] > 0):
            raise ValueError("Matrix must have positive diagonal if constrain is pos_diag_lower_triangular")

    def _get_params(self) -> np.ndarray:
        n, m = self.shape
        # strictly lower-triangular indices
        lower_idx = np.tril_indices(n, k=-1, m=m)
        lower_params = self.matrix[lower_idx]
        # diagonal indices
        diag_idx = np.diag_indices(min(n, m))
        diag_params = softplus_inv(self.matrix[diag_idx])  # invert softplus to get free param
        # concatenate: lower-triangular first, then diagonal
        return np.concatenate([lower_params, diag_params])

        # params = self.get_vectorized(order='F')
        #
        # diag_idx = np.diag_indices(min(n, m))
        # diag_params = softplus_inv(self.matrix[diag_idx].copy())  # invert softplus to get free param
        # params[np.arange(0, min(n, m)) * (n+1)] = diag_params
        #
        # arr = np.ones((n*m,), dtype=bool)
        # for i in range(1, m):
        #     for j in range(i):
        #         arr[i*n+j] = False
        # params = params[arr]
        # return params

        # params = self.get_vectorized(order='C')
        # diag_idx = np.diag_indices(min(n, m))
        # diag_params = softplus_inv(self.matrix[diag_idx].copy())  # invert softplus to get free param
        # params[np.arange(0, min(n, m)) * (m+1)] = diag_params
        # arr = np.ones((n*m,), dtype=bool)
        # for i in range(1, min(n,m)):
        #     for j in range(i):
        #         arr[i+j*m] = False
        # params = params[arr]
        # return params

    def _update_params(self, params) -> None:
        if self.matrix is None:
            matrix = np.zeros(self.shape)
            matrix[np.diag_indices(min(*self.shape))] = 1
            self.matrix = matrix

        n, m = self.shape

        num_lower = m * n - (m + 1) * m // 2 if m < n else n * (n - 1) // 2

        # Split parameters
        lower_params = params[:num_lower]
        diag_params = params[num_lower:]
        # self._matrix = np.diag(softplus(diag_params))

        # Fill strictly lower triangle using a flat view
        idx = 0
        for i in range(1, n):
            row_len = min(i, m)
            self.matrix[i, :row_len] = lower_params[idx:idx + row_len]
            idx += row_len

        # Fill diagonal
        diag_idx = np.diag_indices(min(n, m))
        self._matrix[diag_idx] = softplus(diag_params)

    def _get_row_constraint(self, idx: int):
        n, m = self.shape

        zero_idxs = np.arange(idx + 1, m)

        if idx > m - 1:
            return None, None, None, None

        A = np.zeros((len(zero_idxs), m))
        for i, idx_ in enumerate(zero_idxs):
            A[i, idx_] = 1
        b = np.zeros(len(zero_idxs))

        A_ineq = np.zeros((1, m))
        A_ineq[0, idx] = 1
        b_ineq = np.zeros((1,))

        return A, b.reshape(-1, 1), A_ineq, b_ineq.reshape(-1, 1)

        # if not (m > idx + 2):
        #     return None, None, None, None
        #
        # if idx == m:
        #     A = np.zeros((1, m))
        #     A[0, m] = 1
        #     b = np.array([1])
        #     return A, b.reshape(-1, 1), None, None
        #
        # zero_idxs = np.arange(idx + 1, m)
        # A = np.zeros((len(zero_idxs), m))
        # for i, idx_ in enumerate(zero_idxs):
        #     A[i, idx_] = 1
        # b = np.zeros(len(zero_idxs))
        #
        # A_ineq = np.zeros((1, m))
        # A_ineq[0, idx] = 1
        # b_ineq = np.zeros((1,))
        #
        # return A, b.reshape(-1, 1), A_ineq, b_ineq.reshape(-1, 1)

    def _get_col_constraint(self, idx: int):
        n, m = self.shape

        zero_idxs = np.arange(0, min(n ,idx))

        A = np.zeros((len(zero_idxs), n))
        for i, idx_ in enumerate(zero_idxs):
            A[i, idx_] = 1
        b = np.zeros(len(zero_idxs))

        if idx < n:
            A_ineq = np.zeros((1, n))
            A_ineq[0, idx] = 1
            b_ineq = np.zeros((1,))

            return A, b.reshape(-1, 1), A_ineq, b_ineq.reshape(-1, 1)

        else:
            return A, b.reshape(-1, 1), None, None

    def _jacobian_vecF_params(self):
        n, m = self.shape
        arr = np.arange(n*m).reshape((n,m), order="F")

        lower_idx = np.tril_indices(n, k=-1, m=m)
        diag_idx = np.diag_indices(min(n, m))

        num_lower = len(lower_idx[0])
        j = np.zeros((n*m, self.n_params))
        # for i, idx in enumerate(arr[lower_idx]):
        #     j[idx, i] = 1
        idx = arr[lower_idx]
        j[idx, np.arange(idx.size)] = 1

        derivatives = sigmoid_softplus_inv(np.diagonal(self.matrix))
        # for i, idx in enumerate(arr[diag_idx]):
        #     j[idx, i+num_lower] = derivatives[i]
        idx = arr[diag_idx]
        cols = np.arange(idx.size) + num_lower
        j[idx, cols] = derivatives

        return j

    def _jacobian_vecC_params(self):
        n, m = self.shape
        arr = np.arange(n*m).reshape((n,m), order="C")

        lower_idx = np.tril_indices(n, k=-1, m=m)
        diag_idx = np.diag_indices(min(n, m))

        num_lower = len(lower_idx[0])
        j = np.zeros((n*m, self.n_params))
        for i, idx in enumerate(arr[lower_idx]):
            j[idx, i] = 1

        derivatives = sigmoid_softplus_inv(np.diagonal(self.matrix))
        for i, idx in enumerate(arr[diag_idx]):
            j[idx, i+num_lower] = derivatives[i]

        return j

    @property
    def _n_params(self):
        if self.has_shape:
            n, m = self.shape
            if m < n:
                return m * n - (m - 1) * m // 2
            else:
                return n * (n + 1) // 2
        else:
            raise ValueError("Matrix has no defined shape. Unknown number of parameters")


# -----------
# Symmetry
#----------

class SymmetricMatrix(ConstrainedMatrix):
    _is_elementwise = False
    _enforce_square = True

    def __init__(self, shape: tuple[int, int]):
        if shape[0] != shape[1]:
            raise ValueError("Shape of matrix must be square")

        super().__init__(shape)

    def _check_matrix_constrain(self, matrix):
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Matrix must be symmetric")

    def _get_params(self) -> np.ndarray:
        n, m = self.shape
        idx = np.tril_indices(n, k=0, m=m)
        return self.matrix[idx]

    def _update_params(self, params) -> None:
        if self.matrix is None:
            self.matrix = np.zeros(self.shape)
        n, m = self.shape
        # idx = np.tril_indices(n, k=0, m=m)
        # self._matrix[idx] = params
        # upper_idx = np.triu_indices(n, k=1, m=m)
        # self._matrix[upper_idx] = self._matrix.T[upper_idx]

        idx_shift = 0
        for i in range(n):
            _range = range(0, i + 1)
            self._matrix[i, _range] = params[idx_shift:idx_shift + len(_range)]
            self._matrix[_range, i] = params[idx_shift:idx_shift + len(_range)]
            idx_shift += len(_range)

    def jacobian_vec_params(self, order: Literal['F', 'C'] = 'C'):
        n, m = self.shape
        # n_params = self.n_params

        # Precomputed mapping arr[i, j] -> flat index into vecF (column-major)
        arr = np.arange(n * m).reshape((n, m), order=order)

        # Lower-triangular indices (including diagonal)
        idx_i, idx_j = np.tril_indices(n, m=m)

        # Flattened indices of LT elements
        lt_flat = arr[idx_i, idx_j]
        k = lt_flat.size  # = n_params

        # Allocate Jacobian
        j = np.zeros((n * m, k))

        # Fast fill of lower triangle:
        # Instead of fancy indexing, use broadcasting-like identity placement
        j[lt_flat] = np.eye(k, dtype=j.dtype)

        # ---- Symmetric copy (FAST) ----
        # Upper triangle indices (excluding diagonal)
        ut_i, ut_j = np.triu_indices(n, k=1, m=m)

        # Symmetric LT partner positions (swapped)
        sym_flat = arr[ut_j, ut_i]
        ut_flat = arr[ut_i, ut_j]

        # Fast row copy: slices instead of fancy indexing
        j[ut_flat] = j[sym_flat]

        return j

    @property
    def _n_params(self):
        if self.has_shape:
            n, m = self.shape
            if m < n:
                return m * n - (m - 1) * m // 2
            else:
                return n * (n + 1) // 2
        else:
            raise ValueError("Matrix has no defined shape. Unknown number of parameters")


class SetDiagSymmetricConstraint(ConstrainedMatrix):
    _is_elementwise = False
    _enforce_square = True
    
    def __init__(self, shape: tuple[int, int], diag_value):
        if shape[0] != shape[1]:
            raise ValueError("Shape of matrix must be square")

        super().__init__(shape)

        self._diag_value = diag_value

    def _check_matrix_constrain(self, matrix):
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Matrix must be symmetric")

        idx_diag = np.diag_indices(min(*self.shape))
        if not np.allclose(matrix[idx_diag], self._diag_value):
            raise ValueError("Matrix must have unit diagonal if constrain is unit_diag_lower_triangular")

    def _update_params(self, params) -> None:
        if self.matrix is None:
            matrix = np.zeros(self.shape)
            matrix[np.diag_indices(min(*self.shape))] = self._diag_value
            self.set_matrix_trusted(matrix)
        n, m = self.shape
        idx = np.tril_indices(n, k=-1, m=m)
        self._matrix[idx] = params
        self._matrix.T[idx] = params

    def _get_params(self) -> np.ndarray:
        n, m = self.shape
        idx = np.tril_indices(n, k=-1, m=m)
        return self.matrix[idx]

    def jacobian_vec_params(self, order: Literal['F', 'C'] = 'C'):
        n, m = self.shape
        # n_params = self.n_params

        # Precomputed mapping arr[i, j] -> flat index into vecF (column-major)
        arr = np.arange(n * m).reshape((n, m), order=order)

        # Lower-triangular indices (including diagonal)
        idx_i, idx_j = np.tril_indices(n, k=-1, m=m)

        # Flattened indices of LT elements
        lt_flat = arr[idx_i, idx_j]
        k = lt_flat.size  # = n_params

        # Allocate Jacobian
        j = np.zeros((n * m, k))

        # Fast fill of lower triangle:
        # Instead of fancy indexing, use broadcasting-like identity placement
        j[lt_flat] = np.eye(k, dtype=j.dtype)

        # ---- Symmetric copy (FAST) ----
        # Upper triangle indices (excluding diagonal)
        ut_i, ut_j = np.triu_indices(n, k=1, m=m)

        # Symmetric LT partner positions (swapped)
        sym_flat = arr[ut_j, ut_i]
        ut_flat = arr[ut_i, ut_j]

        # Fast row copy: slices instead of fancy indexing
        j[ut_flat] = j[sym_flat]

        return j

    @property
    def _n_params(self):
        if self.has_shape:
            n, m = self.shape
            if m < n:
                return m * n - (m + 1) * m // 2
            else:
                return n * (n - 1) // 2
        else:
            raise ValueError("Matrix has no defined shape. Unknown number of parameters")


class UnitDiagSymmetricMatrix(SetDiagSymmetricConstraint):

    def __init__(self, shape):
        super().__init__(shape, 1)


class ZeroDiagSymmetricMatrix(SetDiagSymmetricConstraint):

    def __init__(self, shape):
        super().__init__(shape, 0)


class SkewSymmetricMatrix(ConstrainedMatrix):
    _is_elementwise = False
    _enforce_square = True
    
    def _check_matrix_constrain(self, matrix):
        if not np.allclose(matrix, -matrix.T):
            raise ValueError("Matrix must be skew symmetric")

    def _get_params(self) -> np.ndarray:
        n, m = self.shape
        idx = np.tril_indices(n, k=-1, m=m)
        return self.matrix[idx]

    def _update_params(self, params) -> None:
        if self.matrix is None:
            self.set_matrix_trusted(np.zeros(self.shape))
        n, m = self.shape
        # OPTION 1
        # idx = np.tril_indices(n, k=-1, m=m)
        # self._matrix[idx] = params
        # upper_idx = np.triu_indices(n, k=1, m=m)
        # self._matrix[upper_idx] = -self._matrix.T[upper_idx]
        # OPTION 2
        idx_shift = 0
        for i in range(n):
            _range = range(i + 1, n)
            self._matrix[i, _range] = -params[idx_shift:idx_shift+len(_range)]
            self._matrix[_range, i] = params[idx_shift:idx_shift+len(_range)]
            idx_shift += len(_range)

    def jacobian_vec_params(self, order: Literal['F', 'C'] = 'C'):
        n, m = self.shape
        # n_params = self.n_params

        # Precomputed mapping arr[i, j] -> flat index into vecF (column-major)
        arr = np.arange(n * m).reshape((n, m), order=order)

        # Lower-triangular indices (including diagonal)
        idx_i, idx_j = np.tril_indices(n, k=-1, m=m)

        # Flattened indices of LT elements
        lt_flat = arr[idx_i, idx_j]
        k = lt_flat.size  # = n_params

        # Allocate Jacobian
        j = np.zeros((n * m, k))

        # Fast fill of lower triangle:
        # Instead of fancy indexing, use broadcasting-like identity placement
        j[lt_flat] = np.eye(k, dtype=j.dtype)

        # ---- Symmetric copy (FAST) ----
        # Upper triangle indices (excluding diagonal)
        ut_i, ut_j = np.triu_indices(n, k=1, m=m)

        # Symmetric LT partner positions (swapped)
        sym_flat = arr[ut_j, ut_i]
        ut_flat = arr[ut_i, ut_j]

        # Fast row copy: slices instead of fancy indexing
        j[ut_flat] = -j[sym_flat]

        return j

    @property
    def _n_params(self):
        if self.has_shape:
            n, m = self.shape
            if m < n:
                return m * n - (m + 1) * m // 2
            else:
                return n * (n - 1) // 2
        else:
            raise ValueError("Matrix has no defined shape. Unknown number of parameters")

# -----------
# Special Orthogonal
#----------


class SpecialOrthogonalMatrix(ConstrainedMatrixWithUnderlying):

    _is_elementwise = False
    _enforce_square = True
    
    def __init__(self, shape: tuple[int, int]):
        underlying = SkewSymmetricMatrix(shape)
        super().__init__(shape, underlying, store_both=False, run_check_on='underlying')

    def _check_matrix_constrain(self, matrix):
        atol = 1e-8
        is_othogonal = np.allclose(matrix.T @ matrix, np.eye(self.shape[0]), atol=atol)
        if not is_othogonal:
            raise ValueError("Matrix must be orthogonal")
        is_SO = np.isclose(np.linalg.det(matrix), 1.0, atol=atol)
        if not is_SO:
            raise ValueError("Matrix must be orthogonal in SO(n)")

    def _compute_matrix_from_underlying(self) -> np.ndarray:
        return expm(self._underlying.matrix)
    
    def _compute_underlying_from_matrix(self, matrix):
        return logm(matrix)

    def _on_shape_setter(self, shape: tuple):
        self._underlying.shape = shape


SOMatrix = SpecialOrthogonalMatrix

# -----------
# Correlation and Covariance
#----------

STDMatrix = PosDiagLowerTriMatrix


class CovMatrix(ConstrainedMatrixWithUnderlying):

    _is_elementwise = False
    _enforce_square = True

    def __init__(self, shape: tuple[int, int]):
        underlying = STDMatrix(shape)
        super().__init__(shape, underlying, store_both=False, run_check_on='underlying')

    def _compute_matrix_from_underlying(self) -> np.ndarray:
        return self._underlying.matrix @ self._underlying.matrix.T

    def _compute_underlying_from_matrix(self, matrix):
        return np.linalg.cholesky(matrix)

    def _check_matrix_constrain(self, matrix):
        """
        This method should never be called because all constraint checks
        are performed on the underlying matrix.
        """
        raise RuntimeError(
            f"{type(self).__name__} does not perform matrix constraint checks; "
            "they are handled by the underlying matrix."
        )

    def _on_shape_setter(self, shape: tuple):
        self._underlying.shape = shape


class SqrtCorrMatrix(ConstrainedMatrixWithUnderlying):
    _is_elementwise = False
    _enforce_square = True

    def __init__(self, shape: tuple[int, int]):
        dof = shape[0]
        if dof is not None:
            dof = (dof - 1) * dof // 2  # integer division
        underlying = FreeMatrix((dof, 1))
        super().__init__(shape, underlying, store_both=True, run_check_on='top')

    def _compute_matrix_from_underlying(self) -> np.ndarray:
        n = self.shape[0]
        L = np.zeros((n, n))
        params = self._underlying.matrix[:, 0]
        idx = 0
        for i in range(n):
            if i == 0:
                L[i, i] = 1.0
            else:
                # compute row i using hyperspherical coordinates
                # angles for row i: params[idx: idx + i]
                angles = params[idx: idx + i]
                idx += i

                # fill the i-th row
                row = np.zeros(i + 1)
                prod_sin = 1.0
                for j in range(i):
                    row[j] = prod_sin * np.sin(angles[j])
                    prod_sin *= np.cos(angles[j])
                row[i] = prod_sin  # last entry
                L[i, :i + 1] = row
        return L

    def _compute_underlying_from_matrix(self, matrix):
        L = self._matrix
        params = []
        n = self.shape[0]
        for i in range(1, n):
            row_params = []
            for j in range(i):
                # Compute angle using arcsin of the current entry divided by cumulative product of previous cosines
                prod_cos = 1.0
                for k in range(j):
                    prod_cos *= np.cos(row_params[k])
                angle = np.arcsin(L[i, j] / prod_cos)
                row_params.append(angle)
            params.extend(row_params)
        return np.array(params)

    def _check_matrix_constrain(self, matrix):
        """
        Checks that the Cholesky factor L produces a valid correlation matrix.
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix must be a numpy array")
        if matrix.shape != self.shape:
            raise ValueError(f"Matrix must be {self.shape}")

        corr = matrix @ matrix.T
        if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
            raise ValueError("Resulting correlation matrix diagonal is not 1")
        if not np.allclose(corr, corr.T, atol=1e-8):
            raise ValueError("Resulting correlation matrix is not symmetric")
        eigvals = np.linalg.eigvalsh(corr)
        if np.any(eigvals <= 0):
            raise ValueError("Resulting correlation matrix is not positive definite")

    def _on_shape_setter(self, shape: tuple):
        dof = shape[0]
        if dof is not None:
            dof = (dof - 1) * dof // 2  # integer division
        self._underlying.shape = (dof, 1)


class LogCorrMatrix(ConstrainedMatrix):  # Could use a symmetric matrix as underlying but this is slightly easier i think
    _is_elementwise = False
    _enforce_square = True

    def __init__(self, shape: tuple[int, int], warm_start: bool = False):
        super().__init__(shape)
        self._warm_start = warm_start

    @property
    def _n_params(self) -> int:
        if self.has_shape:
            n = self.shape[0]
            return (n - 1) * n // 2
        else:
            raise ValueError("Matrix has no defined shape. Unknown number of parameters")

    def _update_params(self, params) -> None:
        if self.matrix is None:
            matrix = np.zeros(self.shape)
            self.set_matrix_trusted(matrix)
        n, m = self.shape
        idx = np.tril_indices(n, k=-1, m=m)
        self._matrix[idx] = params
        self._matrix.T[idx] = params
        self._matrix = self._calc_matrix_diagonal(self._matrix)

    def _calc_matrix_diagonal(self, matrix):
        if not self._warm_start:
            np.fill_diagonal(self._matrix, 0)

        diag_vec = np.diagonal(matrix)

        tol_value = 1e-8
        n = self.shape[0]

        conv = np.inf
        iter_number = 0
        while conv > np.sqrt(n) * tol_value:
            diag_delta = logm(np.diagonal(expm(matrix)))
            diag_vec = diag_vec - diag_delta
            np.fill_diagonal(matrix, diag_vec)
            conv = np.linalg.norm(diag_delta)
            iter_number += 1

        return matrix

    def _get_params(self) -> np.ndarray:
        idx = np.tril_indices(n=self.shape[0], k=-1, m=self.shape[1])
        return self._matrix[idx]

    def _check_matrix_constrain(self, matrix):
        matrix = expm(matrix)
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix must be a numpy array")
        if matrix.shape != self.shape:
            raise ValueError(f"Matrix must be {self.shape}")

        if not np.allclose(np.diag(matrix), 1.0, atol=1e-8):
            raise ValueError("Resulting correlation matrix diagonal is not 1")
        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ValueError("Resulting correlation matrix is not symmetric")
        eigvals = np.linalg.eigvalsh(matrix)
        if np.any(eigvals < -1e-12):
            raise ValueError("Resulting correlation matrix is not positive definite")


class CorrMatrix(ConstrainedMatrixWithUnderlying):
    _is_elementwise = False
    _enforce_square = True

    def __init__(self, shape: tuple[int | None, int | None], parameterization: Literal['hyperspherical', 'log']):

        self._parameterization = parameterization
        if parameterization == 'hyperspherical':
            underlying = SqrtCorrMatrix(shape)
        elif parameterization == 'log':
            underlying = LogCorrMatrix(shape)
        else:
            raise ValueError(f"Correlation matrix parameterization must be log or hyperspherical")

        super().__init__(shape, underlying, store_both=False, run_check_on='top')


    def _compute_underlying_from_matrix(self, matrix):
        if self._parameterization == 'hyperspherical':
            return np.linalg.cholesky(matrix)
        elif self._parameterization == 'log':
            lm = logm(matrix)
            return np.real_if_close(lm, tol=1e-12)
        else:
            raise ValueError(f"Correlation matrix parameterization must be log or hyperspherical")

    def _compute_matrix_from_underlying(self) -> np.ndarray:
        if self._parameterization == 'hyperspherical':
            return self._underlying.matrix @ self._underlying.matrix.T
        elif self._parameterization == 'log':
            # Use expm and ensure real part
            em = expm(self._underlying.matrix)
            return np.real_if_close(em, tol=1e-12)
        else:
            raise ValueError(f"Correlation matrix parameterization must be log or hyperspherical")

    def _on_shape_setter(self, shape: tuple):
        self._underlying.shape = shape

    def _check_matrix_constrain(self, matrix):
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix must be a numpy array")
        if matrix.shape != self.shape:
            raise ValueError(f"Matrix must be {self.shape}")

        if not np.allclose(np.diag(matrix), 1.0, atol=1e-8):
            raise ValueError("Resulting correlation matrix diagonal is not 1")
        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ValueError("Resulting correlation matrix is not symmetric")
        eigvals = np.linalg.eigvalsh(matrix)
        if np.any(eigvals < -1e-12):
            raise ValueError("Resulting correlation matrix is not positive definite")


# ----------
# Testing
#--------


if __name__ == "__main__":
    import time


    def get_randos(n, m):

        randos = []

        random_free = 10 * (np.random.rand(n, m) - 0.5)
        randos.append(random_free)

        random_utw = random_free.copy()
        random_utw[0, :] = 1
        randos.append(random_utw)

        random_diag = np.zeros((n, m))
        idx = np.diag_indices(min(n, m))
        random_diag[idx] = 10 * (np.random.rand(min(n, m)) - 0.5)
        randos.append(random_diag)

        random_pos_diag = np.zeros((n, m))
        idx = np.diag_indices(min(n, m))
        random_pos_diag[idx] = 10 * np.random.rand(min(n, m))
        randos.append(random_pos_diag)

        random_zero = np.zeros((n, m))
        randos.append(random_zero)

        random_ident = np.zeros((n, m))
        idx = np.diag_indices(min(n, m))
        random_ident[idx] = 1
        randos.append(random_ident)

        random_lt = np.zeros((n, m))
        idx = np.tril_indices(n, k=0, m=m)
        count = int(np.sum(np.ones((n, m))[idx]))
        random_lt[idx] = 10 * (np.random.rand(count) - 0.5)
        randos.append(random_lt)

        random_udlt = np.zeros((n, m))
        idx = np.tril_indices(n, k=-1, m=m)
        count = int(np.sum(np.ones((n, m))[idx]))
        random_udlt[idx] = 10 * (np.random.rand(count))
        idx = np.diag_indices(min(n, m))
        random_udlt[idx] = 1
        randos.append(random_udlt)

        random_zdlt = np.zeros((n, m))
        idx = np.tril_indices(n, k=-1, m=m)
        count = int(np.sum(np.ones((n, m))[idx]))
        random_zdlt[idx] = 10 * (np.random.rand(count))
        idx = np.diag_indices(min(n, m))
        random_zdlt[idx] = 0
        randos.append(random_zdlt)

        random_pdlt = np.zeros((n, m))
        idx = np.tril_indices(n, k=-1, m=m)
        count = int(np.sum(np.ones((n, m))[idx]))
        random_pdlt[idx] = 10 * (np.random.rand(count))
        idx = np.diag_indices(min(n, m))
        random_pdlt[idx] = 10 * (np.random.rand(min(n, m)))
        randos.append(random_pdlt)

        if n == m:
            random_sym = 10 * (np.random.rand(n, m) - 0.5)
            randos.append((random_sym + random_sym.T)/2)

            random_udsym = 10 * (np.random.rand(n, m) - 0.5)
            random_udsym = (random_udsym + random_udsym.T) / 2
            idx = np.diag_indices(min(n, m))
            random_udsym[idx] = 0
            randos.append(random_udsym)

            random_skew_sym = 10 * (np.random.rand(n, m) - 0.5)
            random_skew_sym = (random_skew_sym + random_skew_sym.T) / 2
            idx = np.diag_indices(min(n, m))
            random_skew_sym[idx] = 0
            idx = np.tril_indices(n, k=-1, m=m)
            idx2 = np.triu_indices(n, k=1, m=m)
            random_skew_sym[idx2] = - random_skew_sym.T[idx]
            randos.append(random_skew_sym)

            randos.append(expm(random_skew_sym))

            randos.append(random_pdlt @ random_pdlt.T)

            std = np.sqrt(np.diag(randos[-1]))
            denom = np.outer(std, std)
            corr = randos[-1] / denom
            np.fill_diagonal(corr, 1.0)

            randos.append(corr)

        return randos


    shape0 = (3, 3)
    randoms0 = get_randos(*shape0)

    shape1 = (3, 3)
    randoms1 = get_randos(*shape1)

    matrix_options = [FreeMatrix, UnitTopRowConstrained, DiagonalMatrix, PosDiagonalMatrix, ZeroMatrix, IdentityMatrix,
                      LowerTriangularMatrix, UnitLowerTriMatrix, ZeroLowerTriMatrix, PosDiagLowerTriMatrix,
                      SymmetricMatrix, UnitDiagSymmetricMatrix, SkewSymmetricMatrix, SOMatrix,
                      CovMatrix, CorrMatrix]


    i, j = 13, 6

    print((matrix_options[i], matrix_options[j]))

    mat0 = matrix_options[i](shape0)
    mat0.matrix = randoms0[i]

    mat1 = matrix_options[j](shape1)
    mat1.matrix = randoms0[j]


    mat_comb = CompositeConstraintMatrix([mat0, mat1], combine='hstack')

    rnd_mat = np.hstack([randoms0[i], randoms1[j]])

    print(rnd_mat)

    mat_comb.matrix = rnd_mat

    print(mat_comb.get_params())

    # print(mat_comb.jacobian_vec_params())

    #
    # def is_orthogonal(matrix):
    #     atol = 1e-8
    #     is_othogonal = np.allclose(matrix.T @ matrix, np.eye(matrix.shape[0]), atol=atol)
    #     is_SO = np.isclose(np.linalg.det(matrix), 1.0, atol=atol)
    #
    #     print((is_othogonal, is_SO))
    #
    #
    # constrain_options = list(ConstraintType)
    # n, m = 4, 3
    #
    #
    # for i in range(len(constrain_options)):
    # # for i in [12]:
    #
    #     constrain = constrain_options[i]
    #     print(constrain)
    #
    #     rep_random = 100
    #     for _ in range(rep_random):
    #
    #         # randoms = get_randos(n, m)
    #         # matrix = randoms[i]
    #         # is_orthogonal(matrix)
    #
    #
    #         obj = ConstraintFactory.create_constraint(constrain=constrain, shape=(n, m))
    #
    #
    #         params = np.random.rand(obj.n_params) * 10
    #
    #         obj.update_params(params)
    #         matrix = obj.matrix
    #
    #         obj.matrix = obj.matrix
    #
    #         # obj.matrix = matrix
    #         # if len(obj.get_params()) != n_params1:
    #         #     print(constrain)
    #         #     raise RuntimeError(f"wrong number of params")
    #
    #         params = obj.get_params()
    #
    #         obj.update_params(params)
    #
    #         # if not np.allclose(obj.matrix, matrix):
    #         #     raise RuntimeError(f"get param and update param no good")
    #
    #         params = 10 * (np.random.rand(len(params)) - 0.5)
    #         obj.update_params(params)
    #         obj2 = ConstraintFactory.create_constraint(constrain=constrain, shape=(n, m))
    #         obj2.matrix = obj.matrix
    #
    #         if obj2.is_elementwise:
    #             # Eq and Ineq
    #             for idx in range(n):
    #                 row = matrix[idx].reshape((-1, 1))
    #                 A, b, A_ineq, b_ineq = obj.get_row_constraint(idx)
    #                 if A is not None:
    #                     eq = (A @ row - b)[:, 0]
    #                     if any(i != 0 for i in eq):
    #                         raise RuntimeError(f"Row equality constrain failed for {constrain}")
    #                 if A_ineq is not None:
    #                     ineq = (A_ineq @ row - b_ineq)[:, 0]
    #                     if any(i < 0 for i in ineq):
    #                         raise RuntimeError(f"Row inequality constrain failed for {constrain}")
    #
    #             for idx in range(m):
    #                 col = matrix[:, idx].reshape((-1, 1))
    #                 A, b, A_ineq, b_ineq = obj.get_col_constraint(idx)
    #                 if A is not None:
    #                     eq = (A @ col - b)[:, 0]
    #                     if any(i != 0 for i in eq):
    #                         raise RuntimeError(f"Col equality constrain failed for {constrain}")
    #                 if A_ineq is not None:
    #                     ineq = (A_ineq @ col - b_ineq)[:, 0]
    #                     if any(i < 0 for i in ineq):
    #                         raise RuntimeError(f"Col inequality constrain failed for {constrain}")
    #
    #         JF1 = obj2.numerical_jacobian_vec_params(order='F')
    #         JF2 = obj2.jacobian_vec_params(order='F')
    #
    #         if not np.allclose(JF1, JF2):
    #             print(JF1)
    #             print(JF2)
    #             raise RuntimeError(f"Jac F, not bueno {constrain}")
    #
    #         JC1 = obj2.numerical_jacobian_vec_params(order='C')
    #         JC2 = obj2.jacobian_vec_params(order='C')
    #         if not np.allclose(JC1, JC2):
    #             raise RuntimeError(f"Jac C, not bueno {constrain}")
    #
    #     n_iter = 1000
    #     randoms = get_randos(n, m)
    #     matrix = randoms[i]
    #     obj = ConstraintFactory.create_constraint(constrain=constrain, shape=(n, m))
    #     obj.matrix = matrix
    #     start_time = time.time()
    #     for j in range(n_iter):
    #         JF1 = obj2.numerical_jacobian_vec_params(order='F')
    #         JC1 = obj2.numerical_jacobian_vec_params(order='C')
    #     end_time = time.time()
    #     print(f"Numerical elapsed time: {end_time - start_time}")
    #     start_time = time.time()
    #     for j in range(n_iter):
    #         JF1 = obj2.jacobian_vec_params(order='F')
    #         JC1 = obj2.jacobian_vec_params(order='C')
    #     end_time = time.time()
    #     print(f"Analitical elapsed time: {end_time - start_time}")
    #


