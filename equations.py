import numpy as np
from scipy import sparse


class ViscousBurgers:
    def __init__(self, u, nu, d, d2):
        self.X = StateVector([u])
        N = len(u)
        self.M = sparse.eye(N)
        self.L = -nu * d2.matrix
        self.F = lambda X: -X.data * (d @ X.data)

class Wave:
    def __init__(self, u, v, d2):
        N = len(u)
        I, Z = sparse.eye(N), sparse.csr_matrix((N, N))
        self.X = StateVector([u, v])
        self.M = sparse.bmat([[I, Z], [Z, I]])
        self.L = sparse.bmat([[Z, -I], [-d2.matrix, Z]])
        self.F = lambda X: np.zeros_like(X.data)


class SoundWave:
    def __init__(self, u, p, d, rho0, gammap0):
        N = len(u)
        I, Z = sparse.eye(N), sparse.csr_matrix((N, N))
        M00 = rho0 * I if np.isscalar(rho0) else sparse.diags(rho0)
        L10 = gammap0 * d.matrix if np.isscalar(gammap0) else sparse.diags(gammap0) @ d.matrix

        self.X = StateVector([u, p])
        self.M = sparse.bmat([[M00, Z], [Z, I]])
        self.L = sparse.bmat([[Z, d.matrix], [L10, Z]])
        self.F = lambda X: np.zeros_like(X.data)


class ReactionDiffusion:
    def __init__(self, c, d2, c_target, D):
        self.X = StateVector([c])
        self.M, self.L = sparse.eye(len(c)), -D * d2.matrix
        self.F = lambda X: X.data * (c_target - X.data)

