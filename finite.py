import numpy as np
from scipy.special import factorial
from scipy import sparse

def apply_matrix(mat, arr, ax, **kwargs):
    dim = len(arr.shape)
    # Build Einstein signatures
    mat_sig = [dim, ax]
    arr_sig = list(range(dim))
    out_sig = list(range(dim))
    out_sig[ax] = dim
    if sparse.isspmatrix(mat):
        mat = mat.toarray()
    return np.einsum(mat, mat_sig, arr, arr_sig, out_sig, **kwargs)

def reshape_vector(vec, dims=2, ax=-1):
    shape = [1] * dims
    shape[ax] = vec.size
    return vec.reshape(shape)

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

    def dx_array(self, stencil):
        shape = (self.N, len(stencil))
        dx = np.zeros(shape)
        j_min = -np.min(stencil)
        j_max = np.max(stencil)

        padded_values = np.zeros(self.N + j_min + j_max)
        padded_values[j_min:] = np.concatenate([self.values, self.length + self.values[:j_max]])
        if j_min > 0:
            padded_values[:j_min] = self.values[-j_min:] - self.length

        for i in range(self.N):
            dx[i, :] = padded_values[j_min + i + stencil] - padded_values[j_min + i]

        return dx

class UniformNonPeriodicGrid:
    def __init__(self, N, interval):
        """ Non-uniform grid; no grid points at the endpoints of the interval"""
        self.start = interval[0]
        self.end = interval[1]
        self.dx = (self.end - self.start) / (N - 1)
        self.N = N
        self.values = np.linspace(self.start, self.end, N, endpoint=True)
        
class Domain:
    def __init__(self, grids):
        self.dimension = len(grids)
        self.grids = grids
        self.shape = [grid.N for grid in grids]

    def values(self):
        return [reshape_vector(grid.values, self.dimension, i) for i, grid in enumerate(self.grids)]

    def plotting_arrays(self):
        expanded_shape = np.array(self.shape, dtype=np.int) + 1
        return [np.broadcast_to(reshape_vector(np.concatenate((grid.values, [grid.length])), self.dimension, i), expanded_shape)
                for i, grid in enumerate(self.grids)]
        
class Difference:
    def __matmul__(self, other):
        return apply_matrix(self.matrix, other, ax=self.axis)


class DifferenceUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        if stencil_type == 'centered' and convergence_order % 2 != 0:
            raise ValueError("Centered finite difference has even convergence order")
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self._stencil_shape()
        self._make_stencil(grid)
        self._build_matrix(grid)

    def _stencil_shape(self):
        self.dof = self.derivative_order + self.convergence_order
        if self.stencil_type == 'centered':
            self.dof -= (1 - self.dof % 2)
        self.j = np.arange(self.dof) - self.dof // 2

    def _make_stencil(self, grid):
        i = np.arange(self.dof)[:, None]
        j = self.j[None, :]
        S = (j * grid.dx) ** i / factorial(i)
        b = np.zeros(self.dof)
        b[self.derivative_order] = 1.0
        self.stencil = np.linalg.solve(S, b)

    def _build_matrix(self, grid):
        shape = [grid.N] * 2
        matrix = sparse.diags(self.stencil, self.j, shape=shape).tocsr()

        j_min = -np.min(self.j)
        for i in range(j_min):
            matrix[i, -j_min + i:] = self.stencil[:j_min - i]

        j_max = np.max(self.j)
        for i in range(j_max):
            matrix[-j_max + i, :i + 1] = self.stencil[-i - 1:]

        self.matrix = matrix


class DifferenceNonUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self._stencil_shape()
        self._make_stencil(grid)
        self._build_matrix(grid)

    def _stencil_shape(self):
        self.dof = self.derivative_order + self.convergence_order
        self.j = np.arange(self.dof) - self.dof // 2

    def _make_stencil(self, grid):
        dx = grid.dx_array(self.j)
        i = np.arange(self.dof)[None, :, None]
        dx_i = dx[:, None, :]
        S = dx_i ** i / factorial(i)
        b = np.zeros((grid.N, self.dof))
        b[:, self.derivative_order] = 1.0
        self.stencil = np.linalg.solve(S, b)

    def _build_matrix(self, grid):
        shape = [grid.N] * 2
        diags = [self.stencil[slice(-j, None) if j < 0 else slice(None), i] for i, j in enumerate(self.j)]
        matrix = sparse.diags(diags, self.j, shape=shape).tocsr()

        j_min = -np.min(self.j)
        for i in range(j_min):
            matrix[i, -j_min + i:] = self.stencil[i, :j_min - i]

        j_max = np.max(self.j)
        for i in range(j_max):
            matrix[-j_max + i, :i + 1] = self.stencil[-j_max + i, -i - 1:]

        self.matrix = matrix
