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
        self.t = 0
        self.iter = 0
        self.dt = None
        # self.X = StateVector([u, v])
        grid_x,grid_y = domain.grids
        d2x = DifferenceUniformGrid(2, spatial_order, grid_x, 0)
        d2y = DifferenceUniformGrid(2, spatial_order, grid_y, 1)
        dx = DifferenceUniformGrid(1, spatial_order, grid_x, 0)
        dy = DifferenceUniformGrid(1, spatial_order, grid_y, 1)
        class Diffusionx:
            def __init__(self, u, v, nu, d2x):
                self.X = StateVector([u, v], axis=0)
                N = len(u)
                I = sparse.eye(N, N)
                Z = sparse.csr_matrix((N, N))

                M00 = I
                M01 = Z
                M10 = Z
                M11 = I
                self.M = sparse.bmat([[M00, M01],
                                    [M10, M11]])
                L00 = d2x.matrix
                L01 = Z
                L10 = Z
                L11 = d2x.matrix
                self.L = -nu*sparse.bmat([[L00, L01],
                                    [L10, L11]])
        class Diffusiony:
            def __init__(self, u, v, nu, d2y):
                self.X = StateVector([u, v], axis=1)
                N = len(u)
                I = sparse.eye(N, N)
                Z = sparse.csr_matrix((N, N))

                M00 = I
                M01 = Z
                M10 = Z
                M11 = I
                self.M = sparse.bmat([[M00, M01],
                                    [M10, M11]])
                L00 = d2y.matrix
                L01 = Z
                L10 = Z
                L11 = d2y.matrix
                self.L = -nu*sparse.bmat([[L00, L01],
                                    [L10, L11]])
        diffx = Diffusionx(u,v,nu,d2x)
        diffy = Diffusiony(u,v,nu,d2y)
        self.ts_x = CrankNicolson(diffx,0)
        self.ts_y = CrankNicolson(diffy,1)
        class Advection:
            def __init__(self, u, v, dx, dy):
                self.X = StateVector([u, v])
                N = len(u)
                I = sparse.eye(N, N)
                Z = sparse.csr_matrix((N, N))

                M00 = I
                M01 = Z
                M10 = Z
                M11 = I
                self.M = sparse.bmat([[M00, M01],
                                    [M10, M11]])
                self.L = lambda X: 0*X.data
                # print(np.allclose(u,self.X.variables[0]))
                def f(X):
                    # [u,u] matrix X [dudx, dvdx]
                    udup = sparse.kron(sparse.csr_matrix([[1],[1]]),X.data[:N,:])
                    kronx = sparse.kron(sparse.eye(2,2),dx.matrix) @ X.data
                    vdup = sparse.kron(sparse.csr_matrix([[1],[1]]),X.data[N:,:])
                    # krony = sparse.kron(sparse.eye(2,2),dy.matrix) @ X.data
                    krony = sparse.csr_matrix((2*N,N))
                    krony[:N,:] = dy@X.data[:N,:]
                    krony[N:,:] = dy@X.data[N:,:]
                    return -udup.multiply(kronx)-vdup.multiply(krony)
                self.F = f


        self.ts_a = RK22(Advection(u,v,dx,dy))
        pass

    def step(self, dt):
        self.ts_x.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_a.step(dt/2)
        self.ts_a.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/2)
        self.t += dt
        self.iter += 1
        pass
