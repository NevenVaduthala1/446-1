import numpy as np
from scipy.special import factorial
from scipy import sparse

def apply_matrix(mat, arr, axis):
    if sparse.isspmatrix(mat):
        mat = mat.toarray()

    dim = arr.ndim
    axes_in = list(range(dim))
    axes_out = axes_in.copy()
    axes_out[axis] = dim
    return np.einsum(mat, [dim, axis], arr, axes_in, axes_out)

def reshape_vector(vec_data, dim_count=2, axis=-1):
    shape_vec = [1] * dim_count
    shape_vec[axis] = -1
    return vec_data.reshape(shape_vec)

def axindex(axis, index):
    
    result = [slice(None)] * axis  # Prepare list of slices for leading axes
    result.append(index)  # Append the index for the target axis
    
    return tuple(result)

def axslice(axis, start, stop, step=None):
    slice_obj = slice(start, stop, step)  # Create slice object
    return axindex(axis, slice_obj)  # Use axindex to insert slice in the correct axis

class UniformPeriodicGrid:
    def __init__(self, grid_num, grid_len):
        self.values = np.linspace(0, grid_len, grid_num, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = grid_len
        self.N = grid_num

class NonUniformPeriodicGrid:
    def __init__(self, val_list, grid_len):
        self.values = val_list
        self.length = grid_len
        self.N = len(val_list)

    def dx_array(self, stencil_vals):
        jmin, jmax = -np.min(stencil_vals), np.max(stencil_vals)

        # Create padded values array
        padded_vals = np.concatenate([
            self.values[-jmin:] - self.length if jmin > 0 else [],
            self.values,
            self.values[:jmax] + self.length if jmax > 0 else []])

        return padded_vals[jmin + np.arange(self.N)[:, None] + stencil_vals] - padded_vals[jmin + np.arange(self.N)[:, None]]

class UniformNonPeriodicGrid:
    def __init__(self, grid_num, interval_range):
        self.start = interval_range[0]
        self.end = interval_range[1]
        self.dx = (self.end - self.start)/(grid_num-1)
        self.N = grid_num
        self.values = np.linspace(self.start, self.end, grid_num, endpoint=True)

class Domain:
    def __init__(self, grid_list):
        self.dimension = len(grid_list)
        self.grids = grid_list
        self.shape = [grid.N for grid in grid_list]  # Use list comprehension

    def values(self):
        # Use list comprehension and reshape_vector directly
        return [reshape_vector(grid.values, self.dimension, i) for i, grid in enumerate(self.grids)]

    def plotting_arrays(self):
        expanded_shape = np.array(self.shape) + 1
        # Efficient use of np.concatenate and np.broadcast_to in a list comprehension
        return [
            np.broadcast_to(
                reshape_vector(np.concatenate((grid.values, [grid.length])), self.dimension, i),
                expanded_shape
            )
            for i, grid in enumerate(self.grids)
        ]

class Difference:
    def __matmul__(self, other_arr):
        return apply_matrix(self.matrix, other_arr, axis=self.axis)

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
            # cancellation if derivative order is even
            dof = dof - (1 - dof % 2)
            j = np.arange(dof) - dof//2

        self.dof = dof
        self.j = j

    def _make_stencil(self, grid):

        # assume constant grid spacing
        self.dx = grid.dx
        i = np.arange(self.dof)[:, None]
        j = self.j[None, :]
        S = 1/factorial(i)*(j*self.dx)**i

        b = np.zeros( self.dof )
        b[self.derivative_order] = 1.

        self.stencil = np.linalg.solve(S, b)

    def _build_matrix(self, grid):
        shape = [grid.N] * 2
        matrix = sparse.diags(self.stencil, self.j, shape=shape)
        matrix = matrix.tocsr()
        jmin = -np.min(self.j)
        if jmin > 0:
            for i in range(jmin):
                if isinstance(grid, UniformNonPeriodicGrid):
                    # S[ii,jj] = (jj-i)^ii/(ii)!
                    S = np.zeros((self.dof,self.dof))
                    for ii in range(self.dof):
                        for jj in range(self.dof):
                            S[ii,jj] = 1/factorial(ii)*((jj-i)*self.dx)**ii
                    b = np.zeros(self.dof)
                    b[self.derivative_order] = 1.
                    # nonperiodic stencil
                    npstencil = np.linalg.solve(S,b)
                    matrix[i,0:self.dof] = npstencil
                else:
                    matrix[i,-jmin+i:] = self.stencil[:jmin-i]

        jmax = np.max(self.j)
        if jmax > 0:
            for i in range(jmax):
                if isinstance(grid, UniformNonPeriodicGrid):
                    S = np.zeros((self.dof,self.dof))
                    for ii in range(self.dof):
                        for jj in range(self.dof):
                            S[ii,jj] = 1/factorial(ii)*(((grid.N-self.dof+jj)-(grid.N-jmax+i))*self.dx)**ii
                    b = np.zeros(self.dof)
                    b[self.derivative_order] = 1.
                    # nonperiodic stencil
                    npstencil = np.linalg.solve(S,b)
                    matrix[-jmax+i,-self.dof:] = npstencil
                else:
                    matrix[-jmax+i,:i+1] = self.stencil[-i-1:]
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
        self.dx = grid.dx_array(self.j)
        i = np.arange(self.dof)[None, :, None]
        S = (self.dx[:, None, :] ** i) / factorial(i)
        b = np.zeros((grid.N, self.dof))
        b[:, self.derivative_order] = 1
        self.stencil = np.linalg.solve(S, b)

    def _build_matrix(self, grid):
        shape = [grid.N] * 2
        diags = [
            self.stencil[slice(-jj, None) if jj < 0 else slice(None), i]
            for i, jj in enumerate(self.j)
        ]
        matrix = sparse.diags(diags, self.j, shape=shape).tocsr()
        jmin, jmax = -np.min(self.j), np.max(self.j)

        for i in range(jmin):
            matrix[i, -jmin + i:] = self.stencil[i, :jmin - i]
        for i in range(jmax):
            matrix[-jmax + i, :i + 1] = self.stencil[-jmax + i, -i - 1:]

        self.matrix = matrix



class CenteredFiniteDifference(Difference):
    def __init__(self, grid, axis=0):
        self.axis = axis
        dx_inv = 1 / (2 * grid.dx)
        N = grid.N

        # Diagonal values
        diags = [-dx_inv, 0, dx_inv]

        # Create the sparse matrix with periodic boundary conditions
        matrix = sparse.diags(diags, offsets=[-1, 0, 1], shape=(N, N)).tolil()

        # Set periodic boundary conditions directly
        matrix[0, -1] = -dx_inv
        matrix[-1, 0] = dx_inv

        self.matrix = matrix.tocsr()  # Convert to CSR format for efficient arithmetic

class CenteredFiniteDifference4(Difference):
    def __init__(self, grid, axis=0):
        self.axis = axis
        dx_inv = 1 / (12 * grid.dx)
        N = grid.N

        # Diagonal values for 4th-order finite difference
        diags = [dx_inv, -8 * dx_inv, 0, 8 * dx_inv, -dx_inv]

        # Create the sparse matrix with periodic boundary conditions
        matrix = sparse.diags(diags, offsets=[-2, -1, 0, 1, 2], shape=(N, N)).tolil()

        # Set periodic boundary conditions directly
        matrix[0, -2], matrix[0, -1] = dx_inv, -8 * dx_inv
        matrix[1, -1] = dx_inv
        matrix[-2, 0] = -dx_inv
        matrix[-1, 0], matrix[-1, 1] = 8 * dx_inv, -dx_inv

        self.matrix = matrix.tocsr()  # Convert to CSR format for efficient arithmetic
