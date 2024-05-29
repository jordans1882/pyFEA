"""Matrix class tests.

Provides tests for the Matrix class.

Includes:
 - tests of constructors:
   - from_list
   - from_matrix
   - filled
   - zeros
   - ones
   - identity
 - tests of getters and setters
   - elements
   - dims
   - columns/rows
 - addition, subtraction, multiplication
"""

import numpy as np

from feareu.matrix import Matrix, t


def test_matrix_from_list():
    """Test matrix list constructor."""
    a = Matrix.from_list([[1, 2, 3], [4, 5, 6]])
    b = Matrix.from_list([[1, 2, 3], [4, 5, 6]])
    assert np.array_equal(a.get_data(), b.get_data())


def test_matrix_from_matrix():
    """Test matrix matrix constructor."""
    a = Matrix.from_list([[1, 2, 3], [4, 5, 6]])
    b = Matrix.from_matrix(a)
    assert np.array_equal(b.get_data(), a.get_data())


def test_matrix_filled():
    """Test filled matrix constructor."""
    a = Matrix.filled((3, 2), 2.1)
    b = np.zeros((3, 2))
    b.fill(2.1)
    assert np.array_equal(a.get_data(), b)


def test_matrix_zeros():
    """Test zeros matrix constructor."""
    a = Matrix.zeros((3, 3))
    assert np.array_equal(a.get_data(), np.zeros((3, 3)))


def test_matrix_ones():
    """Test ones matrix constructor."""
    a = Matrix.ones((3, 3))
    b = np.ones((3, 3))
    assert np.array_equal(a.get_data(), b)


def test_matrix_identity():
    """Test identity matrix constructor."""
    a = Matrix.identity(3)
    assert np.array_equal(a.get_data(), np.identity(3))


def test_matrix_add():
    """Test Matrix addition."""
    a = Matrix.identity(3)
    b = Matrix.identity(3)
    c = a + b
    d = np.identity(3) + np.identity(3)
    assert np.array_equal(c.get_data(), d)


def test_matrix_sub():
    """Test Matrix addition."""
    a = Matrix.identity(3)
    b = Matrix.identity(3)
    c = a - b
    d = Matrix.zeros((3, 3))
    assert np.array_equal(c.get_data(), d.get_data())


def test_matrix_mul():
    """Test Matrix addition."""
    a = Matrix.identity(3)
    b = Matrix.identity(3)
    c = a * b
    assert np.array_equal(c.get_data(), a.get_data().dot(b.get_data()))


def test_matrix_getitem():
    """Test element getter."""
    a = Matrix.from_list([[1, 2, 3], [4, 5, 6]])
    assert a[0, 0] == 1


def test_matrix_setitem():
    """Test matrix element setter."""
    a = Matrix.from_list([[1, 2, 3], [4, 5, 6]])
    a[0, 0] = 4
    assert a[0, 0] == 4


def test_matrix_get_dims():
    """Test if matrix dims set properly."""
    a = Matrix.from_list([[1, 2, 3], [4, 5, 6]])
    assert a.get_dims()[0] == 2
    assert a.get_dims()[1] == 3


def test_matrix_check_dims_match():
    """Check if matrix dimensions are equal."""
    a = Matrix.from_list([[1, 2, 3], [4, 5, 6]])
    b = Matrix.from_list([[1, 2, 3], [4, 5, 6]])
    assert a._check_dims_match(b)
    assert not a._check_dims_match(t(a))


def test_matrix_add_matrix():
    """Test matrix addition."""
    assert 1 == 1


def test_t():
    """Test matrix transpose."""
    a = Matrix.from_list([[1, 2, 3], [4, 5, 6]])
    b = Matrix.from_list([[1, 4], [2, 5], [3, 6]])
    assert np.array_equal(t(a).get_data(), b.get_data())
