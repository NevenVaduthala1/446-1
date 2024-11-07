import numpy as np
from scipy import sparse
from timesteppers import StateVector, CrankNicolson, RK22
import finite

class ViscousBurgers:
    def __init__(self, u, nu, d, d2):
        self.u, self.X = u, StateVector([u])
        N = len(u)
        self.M = sparse.eye(N)
        self.L = -nu * d2.matrix
        self.F = lambda X: -X.data * (d @ X.data)


class Wave:
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I, Z = sparse.eye(N), sparse.csr_matrix((N, N))
        self.M = sparse.bmat([[I, Z], [Z, I]])
        self.L = sparse.bmat([[Z, -I], [-d2.matrix, Z]])
        self.F = lambda X: 0 * X.data


class SoundWave:
    def __init__(self, u, p, d, rho0, gammap0):
        self.X = StateVector([u, p])
        N = len(u)
        I, Z = sparse.eye(N), sparse.csr_matrix((N, N))

        M00 = rho0 * I if np.isscalar(rho0) else sparse.diags(rho0)
        self.M = sparse.bmat([[M00, Z], [Z, I]])

        L10 = gammap0 * d.matrix if np.isscalar(gammap0) else sparse.diags(gammap0) @ d.matrix
        self.L = sparse.bmat([[Z, d.matrix], [L10, Z]])

        self.F = lambda X: 0 * X.data

class ReactionDiffusion:
    def __init__(self, c, d2, c_target, D):
        self.X = StateVector([c])
        self.M = sparse.eye(len(c))
        self.L = -D * d2.matrix
        self.F = lambda X: X.data * (c_target - X.data)

class ReactionDiffusion2D:
    def __init__(self, c, D, dx2, dy2):
        self.t = self.iter = self.dt = 0

        # Initialize diffusion time-steppers
        self.ts_x = CrankNicolson(
            type('', (), {
                'X': StateVector([c], axis=0),
                'M': sparse.eye(c.shape[0]),
                'L': -D * dx2.matrix
            })(), axis=0
        )
        self.ts_y = CrankNicolson(
            type('', (), {
                'X': StateVector([c], axis=1),
                'M': sparse.eye(c.shape[1]),
                'L': -D * dy2.matrix
            })(), axis=1
        )

        # Initialize reaction time-stepper with F as a static method
        self.ts_c = RK22(
            type('', (), {
                'X': StateVector([c]),
                'M': sparse.eye(c.size),
                'F': staticmethod(lambda X: X.data * (1 - X.data))
            })()
        )

    def step(self, dt):
        half_dt = dt / 2
        # Operator splitting sequence
        for ts in (self.ts_x, self.ts_y, self.ts_c, self.ts_c, self.ts_y, self.ts_x):
            ts.step(half_dt)
        self.t += dt
        self.iter += 1

class ViscousBurgers2D:
    def __init__(self, u, v, nu, spatial_order, domain):
        self.t, self.iter = 0, 0
        grid_x, grid_y = domain.grids

        # Differential operators
        d2x, d2y = [finite.DifferenceUniformGrid(2, spatial_order, grid, axis) for axis, grid in enumerate(domain.grids)]
        dx, dy = [finite.DifferenceUniformGrid(1, spatial_order, grid, axis) for axis, grid in enumerate(domain.grids)]

        # Function to create a diffusion stepper
        def diffusion_stepper(d2, axis):
            N, I = len(u), sparse.eye(len(u))
            L = -nu * sparse.bmat([[d2.matrix, None], [None, d2.matrix]])
            return CrankNicolson(type('', (), {'X': StateVector([u, v], axis), 'M': sparse.bmat([[I, None], [None, I]]), 'L': L})(), axis)

        self.ts_x, self.ts_y = diffusion_stepper(d2x, 0), diffusion_stepper(d2y, 1)

        # Advection stepper
        class Advection:
            def __init__(self):
                self.X = StateVector([u, v])
                N, I = len(u), sparse.eye(len(u))
                self.M = sparse.bmat([[I, None], [None, I]])
                self.F = lambda X: -(
                    sparse.kron([[1], [1]], X.data[:N]).multiply(sparse.kron(sparse.eye(2), dx.matrix) @ X.data) +
                    sparse.kron([[1], [1]], X.data[N:]).multiply(dy @ X.data)
                )

        self.ts_a = RK22(Advection())

    def step(self, dt):
        half_dt = dt / 2
        for ts in [self.ts_x, self.ts_y, self.ts_a, self.ts_a, self.ts_y, self.ts_x]:
            ts.step(half_dt)
        self.t += dt
        self.iter += 1
