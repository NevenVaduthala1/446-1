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

        # Combined diffusion in x and y directions
        class Diffusion:
            def __init__(self, c, D, d2, axis):
                self.X = StateVector([c], axis=axis)
                N = c.shape[axis]
                self.M = sparse.eye(N)
                self.L = -D * d2.matrix

        diffx = Diffusion(c, D, dx2, axis=0)
        diffy = Diffusion(c, D, dy2, axis=1)

        # Reaction term
        class Reaction:
            def __init__(self, c):
                self.X = StateVector([c])
                self.M = sparse.eye(c.size)
                self.F = lambda X: X.data * (1 - X.data)

        reaction = Reaction(c)

        # Initialize timesteppers
        self.ts_c = RK22(reaction)
        self.ts_x = CrankNicolson(diffx, axis=0)
        self.ts_y = CrankNicolson(diffy, axis=1)

    def step(self, dt):
        half_dt = dt / 2
        self.ts_x.step(half_dt)
        self.ts_y.step(half_dt)
        self.ts_c.step(half_dt)
        self.ts_c.step(half_dt)
        self.ts_y.step(half_dt)
        self.ts_x.step(half_dt)
        self.t += dt
        self.iter += 1

class ViscousBurgers2D:
    def __init__(self, u, v, nu, spatial_order, domain):
        self.t = self.iter = self.dt = 0

        grid_x, grid_y = domain.grids
        dx = finite.DifferenceUniformGrid(1, spatial_order, grid_x, axis=0)
        dy = finite.DifferenceUniformGrid(1, spatial_order, grid_y, axis=1)
        d2x = finite.DifferenceUniformGrid(2, spatial_order, grid_x, axis=0)
        d2y = finite.DifferenceUniformGrid(2, spatial_order, grid_y, axis=1)

        # Function to create diffusion operators
        def create_diffusion(u, v, nu, d2, axis):
            N = u.shape[axis]
            I = sparse.eye(N)
            Z = sparse.csr_matrix((N, N))
            L = -nu * d2.matrix
            M = sparse.bmat([[I, Z], [Z, I]])
            L = sparse.bmat([[L, Z], [Z, L]])
            X = StateVector([u, v], axis=axis)
            return type('', (), {'X': X, 'M': M, 'L': L})()

        # Initialize diffusion timesteppers
        self.ts_x = CrankNicolson(create_diffusion(u, v, nu, d2x, axis=0), axis=0)
        self.ts_y = CrankNicolson(create_diffusion(u, v, nu, d2y, axis=1), axis=1)

        # Advection term
        class Advection:
            def __init__(self, u, v, dx, dy):
                self.X = StateVector([u, v])
                N = u.size
                self.M = sparse.eye(2 * N)

                def f(X):
                    u_data, v_data = np.split(X.data, 2)
                    dudx = dx.matrix @ u_data
                    dvdy = dy.matrix @ v_data
                    dudy = dy.matrix @ u_data
                    dvdx = dx.matrix @ v_data
                    adv_u = u_data * dudx + v_data * dudy
                    adv_v = u_data * dvdx + v_data * dvdy
                    return -np.concatenate([adv_u, adv_v])

                self.F = f

        self.ts_a = RK22(Advection(u, v, dx, dy))

    def step(self, dt):
        half_dt = dt / 2
        for ts in [self.ts_x, self.ts_y, self.ts_a, self.ts_a, self.ts_y, self.ts_x]:
            ts.step(half_dt)
        self.t += dt
        self.iter += 1

