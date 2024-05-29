"""
The Matrix module.

Contains Matrix/Vector classes backended by numpy.

Implements:
  - Printing
  - Addition
  - Subtraction
  - Multiplication
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt


class Matrix:
    """Matrix Class."""

    def __init__(self, data: npt.NDArray):
        """Initialize a Matrix class."""
        self._data = data
        self._m = data.shape[0]
        self._n = data.shape[1]

    @classmethod
    def from_np(cls, data: npt.NDArray):
        """Construct Matrix class from numpy array."""
        return cls(data)

    @classmethod
    def from_list(cls, data: list):
        """Construct Matrix class from list."""
        ret = cls(np.asarray(data))
        return ret

    @classmethod
    def from_matrix(cls, matrix: Matrix):
        """Construct Matrix from Matrix."""
        data = matrix.get_data()
        return cls(data)

    @classmethod
    def filled(cls, dims: tuple, value: float):
        """Construct from dims and float. Fills with value provided."""
        data = np.zeros(dims)
        data.fill(value)
        return cls(data)

    @classmethod
    def zeros(cls, dims: tuple):
        """Construct matrix of zeros with dims provided."""
        data = np.zeros(dims)
        return cls(data)

    @classmethod
    def ones(cls, dims: tuple):
        """Construct matrix of ones with dims provided."""
        data = np.ones(dims)
        return cls(data)

    @classmethod
    def identity(cls, dim: int):
        """Construct from matrix."""
        data = np.identity(dim)
        return cls(data)

    def __getitem__(self, idx: tuple):
        """Get an element of the matrix."""
        return self._data[idx[0]][idx[1]]

    def __setitem__(self, idx: tuple, value: float):
        """Set an alement of the matrix."""
        self._data[idx[0]][idx[1]] = value

    def __add__(self, b: Matrix):
        """Add two matrices together."""
        return self._add_matrix(b)

    def __repr__(self):
        """Print Matrix object."""
        return str(self._data).replace("],", "]\n")

    def __mul__(self, other) -> Matrix:
        """Add two matrices together with + operator."""
        if isinstance(other, Matrix):
            return self._multiply_matrix(other)
        else:
            raise ValueError("Other is not a Matrix! Cannot multiply!")

    def __sub__(self, b: Matrix):
        """Subtract two matrices with - operator."""
        return self._subtract_matrix(b)

    def _check_dims_match(self, other):
        """Check dims match between two matrices."""
        return self.get_dims() == other.get_dims()

    def _add_matrix(self, b: Matrix) -> Matrix:
        """Add matrix to self."""
        if not self._check_dims_match(b):
            raise ValueError("Error: Dims do not match for matrix addition.")
        return Matrix(self.get_data() + b.get_data())

    def _subtract_matrix(self, other: Matrix) -> Matrix:
        if not self._check_dims_match(other):
            raise ValueError(
                "Error: Matrix dims don't match for addition\n "
                f"First matrix has dims {self.get_dims()}\n"
                f"and second has {other.get_dims()}"
            )
        return Matrix(self.get_data() - other.get_data())

    def get_data(self) -> npt.NDArray:
        """Get underlying numpy data."""
        return self._data

    def get_dims(self) -> tuple:
        """Get dimensions of matrix as a dictionary."""
        return (self._m, self._n)

    def get_row(self, row_idx):
        """Get the idx row."""
        return self._data[row_idx]

    def get_column(self, col_idx):
        """Get the idx column"""
        return [row[col_idx] for row in self._data]

    # TODO: change set_dims to a resize() method
    # def set_dims(self, dims: tuple):
    #     """Set matrix dimensions as a dictionary"""
    #     self._m = dims[0]
    #     self._n = dims[1]

    def _multiply_matrix(self, other: Matrix):
        return Matrix(self.get_data().dot(other.get_data()))

    # def _multiply_vector(self, bvec: Vector):
    #     return 87

    # # TODO: implement matrix multiplication dim checker
    # def _mult_dim_check(self, bmat: Matrix):
    #     pass

    def transpose(self) -> Matrix:
        """Transpose self."""
        return Matrix(self._data.transpose())


def t(a: Matrix) -> Matrix:
    """Transpose a matrix."""
    return a.transpose()


# def inv(a: Matrix):
#     """Return the matrix inverse."""
#     return a.inverse()

# a = Matrix.from_list([[1,2,3],[4,5,6]])
# a - t(a)


class Vector:
    """Documentation for Vector."""

    def __init__(self):
        """Initialze a vector."""
        super(Vector, self).__init__()
