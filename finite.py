#CODE

import numpy as np
from scipy.special import factorial
from scipy import sparse

# Helper functions from farray
def apply_matrix(matrix, array, axis, **kw):
    """Contract any direction of a multidimensional array with a matrix."""
    dim = len(array.shape)
    # Build Einstein signatures
    mat_sig = [dim, axis]
    arr_sig = list(range(dim))
    out_sig = list(range(dim))
    out_sig[axis] = dim
    # Handle sparse matrices
    if sparse.isspmatrix(matrix):
        matrix = matrix.toarray()
    return np.einsum(matrix, mat_sig, array, arr_sig, out_sig, **kw)

def reshape_vector(data, dim=2, axis=-1):
    """Reshape 1-dim array as a multidimensional vector."""
    # Build multidimensional shape
    shape = [1] * dim
    shape[axis] = data.size
    return data.reshape(shape)

def axindex(axis, index):
    """Index array along specified axis."""
    if axis < 0:
        raise ValueError("`axis` must be positive")
    # Add empty slices for leading axes
    return (slice(None),)*axis + (index,)

def axslice(axis, start, stop, step=None):
    """Slice array along a specified axis."""
    return axindex(axis, slice(start, stop, step))


# Grid classes
class UniformPeriodicGrid:
    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N

class NonUniformPeriodicGrid:
    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)

    def dx_array(self, j):
        shape = (self.N, len(j))
        dx = np.zeros(shape)
        jmin = -np.min(j)
        jmax = np.max(j)

        values_padded = np.zeros(self.N + jmin + jmax)
        if jmin > 0:
            values_padded[:jmin] = self.values[-jmin:] - self.length
        if jmax > 0:
            values_padded[jmin:-jmax] = self.values
            values_padded[-jmax:] = self.length + self.values[:jmax]
        else:
            values_padded[jmin:] = self.values

        for i in range(self.N):
            dx[i, :] = values_padded[jmin+i+j] - values_padded[jmin+i]

        return dx

class UniformNonPeriodicGrid:
    def __init__(self, N, interval):
        """ Non-uniform grid; no grid points at the endpoints of the interval"""
        self.start = interval[0]
        self.end = interval[1]
        self.dx = (self.end - self.start)/(N-1)
        self.N = N
        self.values = np.linspace(self.start, self.end, N, endpoint=True)


# Domain Class
class Domain:
    def __init__(self, grids):
        self.dimension = len(grids)
        self.grids = grids
        shape = []
        for grid in self.grids:
            shape.append(grid.N)
        self.shape = shape

    def values(self):
        v = []
        for i, grid in enumerate(self.grids):
            grid_v = grid.values
            grid_v = reshape_vector(grid_v, self.dimension, i)
            v.append(grid_v)
        return v

    def plotting_arrays(self):
        v = []
        expanded_shape = np.array(self.shape, dtype=np.int)
        expanded_shape += 1
        for i, grid in enumerate(self.grids):
            grid_v = grid.values
            grid_v = np.concatenate((grid_v, [grid.length]))
            grid_v = reshape_vector(grid_v, self.dimension, i)
            grid_v = np.broadcast_to(grid_v, expanded_shape)
            v.append(grid_v)
        return v


# Difference classes and operators
class Difference:
    def __matmul__(self, other):
        return apply_matrix(self.matrix, other, axis=self.axis)


class DifferenceUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        if stencil_type == 'centered' and convergence_order % 2 != 0:
            raise ValueError("Centered finite difference has even convergence order")

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self._stencil_shape(stencil_type)
        self._make_stencil(grid)
        self._build_matrix(grid)

    def _stencil_shape(self, stencil_type):
        dof = self.derivative_order + self.convergence_order
        if stencil_type == 'centered':
            dof = dof - (1 - dof % 2)  # Ensure it's even
            j = np.arange(dof) - dof // 2
        self.dof = dof
        self.j = j

    def _make_stencil(self, grid):
        self.dx = grid.dx
        i = np.arange(self.dof)[:, None]
        j = self.j[None, :]
        S = 1 / factorial(i) * (j * self.dx)**i

        b = np.zeros(self.dof)
        b[self.derivative_order] = 1.0
        self.stencil = np.linalg.solve(S, b)

    def _build_matrix(self, grid):
        shape = [grid.N] * 2
        matrix = sparse.diags(self.stencil, self.j, shape=shape)
        matrix = matrix.tocsr()

        jmin = -np.min(self.j)
        if jmin > 0:
            for i in range(jmin):
                matrix[i, -jmin+i:] = self.stencil[:jmin-i]

        jmax = np.max(self.j)
        if jmax > 0:
            for i in range(jmax):
                matrix[-jmax+i, :i+1] = self.stencil[-i-1:]

        self.matrix = matrix


class DifferenceNonUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        if (derivative_order + convergence_order) % 2 == 0:
            raise ValueError("The derivative plus convergence order must be odd for centered finite difference")

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self._stencil_shape(stencil_type)
        self._make_stencil(grid)
        self._build_matrix(grid)

    def _stencil_shape(self, stencil_type):
        dof = self.derivative_order + self.convergence_order
        j = np.arange(dof) - dof // 2
        self.dof = dof
        self.j = j

    def _make_stencil(self, grid):
        self.dx = grid.dx_array(self.j)

        i = np.arange(self.dof)[None, :, None]
        dx = self.dx[:, None, :]
        S = 1 / factorial(i) * (dx)**i

        b = np.zeros((grid.N, self.dof))
        b[:, self.derivative_order] = 1.0
        self.stencil = np.linalg.solve(S, b)

    def _build_matrix(self, grid):
        shape = [grid.N] * 2
        diags = []
        for i, jj in enumerate(self.j):
            if jj < 0:
                s = slice(-jj, None, None)
            else:
                s = slice(None, None, None)
            diags.append(self.stencil[s, i])
        matrix = sparse.diags(diags, self.j, shape=shape)
        matrix = matrix.tocsr()

        jmin = -np.min(self.j)
        if jmin > 0:
            for i in range(jmin):
                matrix[i, -jmin+i:] = self.stencil[i, :jmin-i]

        jmax = np.max(self.j)
        if jmax > 0:
            for i in range(jmax):
                matrix[-jmax+i, :i+1] = self.stencil[-jmax+i, -i-1:]

        self.matrix = matrix
