import numpy as np
from scipy.special import factorial
from scipy import sparse

# Helper functions from farray
def apply_matrix(mat_data, arr_data, ax_idx, **kwargs):
    """Contract any direction of a multidimensional array with a matrix."""
    dim_count = len(arr_data.shape)
    # Build Einstein signatures
    mat_sig = [dim_count, ax_idx]
    arr_sig = list(range(dim_count))
    out_sig = list(range(dim_count))
    out_sig[ax_idx] = dim_count
    # Handle sparse matrices
    if sparse.isspmatrix(mat_data):
        mat_data = mat_data.toarray()
    return np.einsum(mat_data, mat_sig, arr_data, arr_sig, out_sig, **kwargs)

def reshape_vector(vec_data, dim_count=2, ax_idx=-1):
    """Reshape 1-dim array as a multidimensional vector."""
    # Build multidimensional shape
    shape_vec = [1] * dim_count
    shape_vec[ax_idx] = vec_data.size
    return vec_data.reshape(shape_vec)

# Grid classes
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
        shape_arr = (self.N, len(stencil_vals))
        dx_matrix = np.zeros(shape_arr)
        min_stencil = -np.min(stencil_vals)
        max_stencil = np.max(stencil_vals)

        padded_vals = np.zeros(self.N + min_stencil + max_stencil)
        if min_stencil > 0:
            padded_vals[:min_stencil] = self.values[-min_stencil:] - self.length
        if max_stencil > 0:
            padded_vals[min_stencil:-max_stencil] = self.values
            padded_vals[-max_stencil:] = self.length + self.values[:max_stencil]
        else:
            padded_vals[min_stencil:] = self.values

        for i in range(self.N):
            dx_matrix[i, :] = padded_vals[min_stencil + i + stencil_vals] - padded_vals[min_stencil + i]

        return dx_matrix

class UniformNonPeriodicGrid:
    def __init__(self, grid_num, interval_range):
        """ Non-uniform grid; no grid points at the endpoints of the interval"""
        self.start = interval_range[0]
        self.end = interval_range[1]
        self.dx = (self.end - self.start) / (grid_num - 1)
        self.N = grid_num
        self.values = np.linspace(self.start, self.end, grid_num, endpoint=True)


# Domain Class
class Domain:
    def __init__(self, grid_list):
        self.dimension = len(grid_list)
        self.grids = grid_list
        shape_list = []
        for grid in self.grids:
            shape_list.append(grid.N)
        self.shape = shape_list

    def values(self):
        val_list = []
        for i, grid in enumerate(self.grids):
            grid_values = grid.values
            grid_values = reshape_vector(grid_values, self.dimension, i)
            val_list.append(grid_values)
        return val_list

    def plotting_arrays(self):
        val_list = []
        expanded_shape = np.array(self.shape, dtype=np.int)
        expanded_shape += 1
        for i, grid in enumerate(self.grids):
            grid_values = grid.values
            grid_values = np.concatenate((grid_values, [grid.length]))
            grid_values = reshape_vector(grid_values, self.dimension, i)
            grid_values = np.broadcast_to(grid_values, expanded_shape)
            val_list.append(grid_values)
        return val_list


# Difference classes and operators
class Difference:
    def __matmul__(self, other_arr):
        return apply_matrix(self.matrix, other_arr, ax_idx=self.axis)


class DifferenceUniformGrid(Difference):
    def __init__(self, deriv_order, conv_order, grid_inst, ax_idx=0, stencil_choice='centered'):
        if stencil_choice == 'centered' and conv_order % 2 != 0:
            raise ValueError("Centered finite difference has even convergence order")

        self.derivative_order = deriv_order
        self.convergence_order = conv_order
        self.stencil_type = stencil_choice
        self.axis = ax_idx
        self._stencil_shape(stencil_choice)
        self._make_stencil(grid_inst)
        self._build_matrix(grid_inst)

    def _stencil_shape(self, stencil_choice):
        dof_size = self.derivative_order + self.convergence_order
        if stencil_choice == 'centered':
            dof_size = dof_size - (1 - dof_size % 2)  # Ensure it's even
            stencil_pos = np.arange(dof_size) - dof_size // 2
        self.dof = dof_size
        self.j = stencil_pos

    def _make_stencil(self, grid_inst):
        self.dx = grid_inst.dx
        i_vals = np.arange(self.dof)[:, None]
        j_vals = self.j[None, :]
        stencil_matrix = 1 / factorial(i_vals) * (j_vals * self.dx)**i_vals

        b_vals = np.zeros(self.dof)
        b_vals[self.derivative_order] = 1.0
        self.stencil = np.linalg.solve(stencil_matrix, b_vals)

    def _build_matrix(self, grid_inst):
        mat_shape = [grid_inst.N] * 2
        mat_instance = sparse.diags(self.stencil, self.j, shape=mat_shape)
        mat_instance = mat_instance.tocsr()

        min_j = -np.min(self.j)
        if min_j > 0:
            for i in range(min_j):
                mat_instance[i, -min_j + i:] = self.stencil[:min_j - i]

        max_j = np.max(self.j)
        if max_j > 0:
            for i in range(max_j):
                mat_instance[-max_j + i, :i + 1] = self.stencil[-i - 1:]

        self.matrix = mat_instance


class DifferenceNonUniformGrid(Difference):
    def __init__(self, deriv_order, conv_order, grid_inst, ax_idx=0, stencil_choice='centered'):
        if (deriv_order + conv_order) % 2 == 0:
            raise ValueError("The derivative plus convergence order must be odd for centered finite difference")

        self.derivative_order = deriv_order
        self.convergence_order = conv_order
        self.stencil_type = stencil_choice
        self.axis = ax_idx
        self._stencil_shape(stencil_choice)
        self._make_stencil(grid_inst)
        self._build_matrix(grid_inst)

    def _stencil_shape(self, stencil_choice):
        dof_size = self.derivative_order + self.convergence_order
        stencil_pos = np.arange(dof_size) - dof_size // 2
        self.dof = dof_size
        self.j = stencil_pos

    def _make_stencil(self, grid_inst):
        self.dx = grid_inst.dx_array(self.j)

        i_vals = np.arange(self.dof)[None, :, None]
        dx_vals = self.dx[:, None, :]
        stencil_matrix = 1 / factorial(i_vals) * (dx_vals)**i_vals

        b_vals = np.zeros((grid_inst.N, self.dof))
        b_vals[:, self.derivative_order] = 1.0
        self.stencil = np.linalg.solve(stencil_matrix, b_vals)

    def _build_matrix(self, grid_inst):
        mat_shape = [grid_inst.N] * 2
        diag_list = []
        for i, j_val in enumerate(self.j):
            if j_val < 0:
                slice_vals = slice(-j_val, None, None)
            else:
                slice_vals = slice(None, None, None)
            diag_list.append(self.stencil[slice_vals, i])
        mat_instance = sparse.diags(diag_list, self.j, shape=mat_shape)
        mat_instance = mat_instance.tocsr()

        min_j = -np.min(self.j)
        if min_j > 0:
            for i in range(min_j):
                mat_instance[i, -min_j + i:] = self.stencil[i, :min_j - i]

        max_j = np.max(self.j)
        if max_j > 0:
            for i in range(max_j):
                mat_instance[-max_j + i, :i + 1] = self.stencil[-max_j + i, -i - 1:]

        self.matrix = mat_instance
