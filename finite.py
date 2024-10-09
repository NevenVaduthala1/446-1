# Helper functions from K. J. Burns
import numpy as np
from scipy import sparse

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
        self.b = b
        
        self.stencil = np.linalg.solve(S, b)
        #print(self.stencil)
    def _build_matrix(self, grid):
        shape = [grid.N] * 2
        matrix = sparse.diags(self.stencil, self.j, shape=shape)
        matrix = matrix.tocsr()
        j = self.j

        
        if isinstance(grid, UniformNonPeriodicGrid):
            rep = len(j)//2 
            
            #top row portion
            s_new = np.zeros([len(j), len(j)])
            for i in range(len(j)):
                for k in range(len(j)):
                    s_new[i,k] = ((k*self.dx)**i)/factorial(i)
            a_s = np.linalg.solve(s_new,self.b)
            matrix[0,:len(j)] = a_s

            #last row

            s_inv = np.copy(s_new[:,::-1])
            
            for i in range(len(j)):
                s_inv[i] *= (-1)**i
            
            last_a_s = np.linalg.solve(s_inv, self.b)
           
            for i in range(len(last_a_s)):
                matrix[-1, -i-1] = last_a_s[-i-1]
            

            # the rest of the rows
            for i in range(1, rep):
                appcol = np.zeros(len(j))
                for jj in range(len(j)):
                    appcol[jj] = ((-i*self.dx)**jj)/factorial(jj)
                

                s_mod = np.zeros((len(j), len(j)))
                s_mod[:,0] = appcol

                s_mod[:,1:] = s_new[:,:-1]
                
                a_s = np.linalg.solve(s_mod, self.b)
                matrix[i,:len(j)] = a_s
                s_new = np.copy(s_mod)
                


                #bottom rows
                s_inv = np.copy(s_mod[:,::-1])
                for k in range(len(j)):
                    s_inv[k,:] *= (-1)**k
                last_a_s = np.linalg.solve(s_inv, self.b)
                for kk in range(len(last_a_s)):
                    matrix[-i-1, -kk-1] = last_a_s[-kk-1]

            

        if isinstance(grid, UniformPeriodicGrid):
            jmin = -np.min(self.j)
            if jmin > 0:
                for i in range(jmin):
                    if isinstance(grid, UniformNonPeriodicGrid):
                        pass
                    else:
                        matrix[i,-jmin+i:] = self.stencil[:jmin-i]

            jmax = np.max(self.j)
            if jmax > 0:
                for i in range(jmax):
                    if isinstance(grid, UniformNonPeriodicGrid):
                        pass
                    else:
                        matrix[-jmax+i,:i+1] = self.stencil[-i-1:]
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
        j = np.arange(dof) - dof//2
        self.dof = dof
        self.j = j

    def _make_stencil(self, grid):
        self.dx = grid.dx_array(self.j)

        i = np.arange(self.dof)[None, :, None]
        dx = self.dx[:, None, :]
        S = 1/factorial(i)*(dx)**i

        b = np.zeros( (grid.N, self.dof) )
        b[:, self.derivative_order] = 1.

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
                matrix[i,-jmin+i:] = self.stencil[i, :jmin-i]

        jmax = np.max(self.j)
        if jmax > 0:
            for i in range(jmax):
                matrix[-jmax+i,:i+1] = self.stencil[-jmax+i, -i-1:]

        self.matrix = matrix


class ForwardFiniteDifference(Difference):

    def __init__(self, grid, axis=0):
        self.axis = axis
        h = grid.dx
        N = grid.N
        j = [0, 1]
        diags = np.array([-1/h, 1/h])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h
        self.matrix = matrix


class CenteredFiniteDifference(Difference):

    def __init__(self, grid, axis=0):
        self.axis = axis
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([-1/(2*h), 0, 1/(2*h)])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/(2*h)
        matrix[0, -1] = -1/(2*h)
        self.matrix = matrix


class CenteredFiniteSecondDifference(Difference):

    def __init__(self, grid, axis=0):
        self.axis = axis
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([1/h**2, -2/h**2, 1/h**2])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h**2
        matrix[0, -1] = 1/h**2
        self.matrix = matrix


class CenteredFiniteDifference4(Difference):

    def __init__(self, grid, axis=0):
        self.axis = axis
        h = grid.dx
        N = grid.N
        j = [-2, -1, 0, 1, 2]
        diags = np.array([1, -8, 0, 8, -1])/(12*h)
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-2, 0] = -1/(12*h)
        matrix[-1, 0] = 8/(12*h)
        matrix[-1, 1] = -1/(12*h)

        matrix[0, -2] = 1/(12*h)
        matrix[0, -1] = -8/(12*h)
        matrix[1, -1] = 1/(12*h)
        self.matrix = matrix



class BoundaryCondition:

    def __init__(self, derivative_order, convergence_order, grid):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.dof = self.derivative_order + self.convergence_order
        self.grid = grid
        self._build_vector()

    def _coeffs(self, dx, j):
        i = np.arange(self.dof)[:, None]
        j = j[None, :]
        S = 1/factorial(i)*(j*dx)**i

        b = np.zeros( self.dof )
        b[self.derivative_order] = 1.

        return np.linalg.solve(S, b)


class Left(BoundaryCondition):

    def _build_vector(self):
        dx = self.grid.dx
        j = np.arange(self.dof)

        coeffs = self._coeffs(dx, j)

        self.vector = np.zeros(self.grid.N)
        self.vector[:self.dof] = coeffs


class Right(BoundaryCondition):

    def _build_vector(self):
        dx = self.grid.dx
        j = np.arange(self.dof) - self.dof + 1

        coeffs = self._coeffs(dx, j)

        self.vector = np.zeros(self.grid.N)
        self.vector[-self.dof:] = coeffs
