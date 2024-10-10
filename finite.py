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

def reshape_vector(vec_data, dim_count=2, ax_idx=-1):
    shape_vec = [1] * dim_count
    shape_vec[ax_idx] = -1  
    return vec_data.reshape(shape_vec)


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


# Domain Class
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

# Difference classes and operators
class Difference:
    def __matmul__(self, other_arr):
        return apply_matrix(self.matrix, other_arr, axis=self.axis) 


class DifferenceUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid_inst, ax_idx=0, stencil_choice='centered'):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_choice
        self.axis = ax_idx
        self._build_stencil(grid_inst)
        self._build_matrix(grid_inst)

    def _build_stencil(self, grid_inst):
        dof_size = self.derivative_order + self.convergence_order
        if self.stencil_type == 'centered':
            dof_size -= (1 - dof_size % 2)  # Ensure it's even
        self.j = np.arange(dof_size) - dof_size // 2  # Stencil positions

        i_vals = np.arange(dof_size)[:, None]
        j_vals = self.j[None, :]
        stencil_matrix = (j_vals * grid_inst.dx) ** i_vals / factorial(i_vals)

        b_vals = np.zeros(dof_size)
        b_vals[self.derivative_order] = 1.0
        self.stencil = np.linalg.solve(stencil_matrix, b_vals)

    def _build_matrix(self, grid_inst):
        mat_instance = sparse.diags(self.stencil, self.j, shape=(grid_inst.N, grid_inst.N)).tocsr()

        # Handle boundary conditions
        jmin, jmax = -np.min(self.j), np.max(self.j)
        for i in range(jmin):
            mat_instance[i, -jmin + i:] = self.stencil[:jmin - i]
        for i in range(jmax):
            mat_instance[-jmax + i, :i + 1] = self.stencil[-i - 1:]

        self.matrix = mat_instance



class DifferenceNonUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid_inst, ax_idx=0, stencil_choice='centered'):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_choice
        self.axis = ax_idx
        self._build_stencil(grid_inst)
        self._build_matrix(grid_inst)

    def _build_stencil(self, grid_inst):
        dof_size = self.derivative_order + self.convergence_order
        self.j = np.arange(dof_size) - dof_size // 2  # Stencil positions
        self.dx = grid_inst.dx_array(self.j)

        # Create stencil matrix using broadcasting
        i_vals = np.arange(dof_size)[None, :, None]
        dx_vals = self.dx[:, None, :]
        stencil_matrix = (dx_vals ** i_vals) / factorial(i_vals)

        b_vals = np.zeros((grid_inst.N, dof_size))
        b_vals[:, self.derivative_order] = 1.0
        self.stencil = np.linalg.solve(stencil_matrix, b_vals)

    def _build_matrix(self, grid_inst):
        mat_instance = sparse.diags(
            [self.stencil[slice(-j, None) if j < 0 else slice(None), i] for i, j in enumerate(self.j)],
            self.j,
            shape=(grid_inst.N, grid_inst.N)
        ).tocsr()

        # Handle boundary conditions
        jmin, jmax = -np.min(self.j), np.max(self.j)
        for i in range(jmin):
            mat_instance[i, -jmin + i:] = self.stencil[i, :jmin - i]
        for i in range(jmax):
            mat_instance[-jmax + i, :i + 1] = self.stencil[-jmax + i, -i - 1:]

        self.matrix = mat_instance
