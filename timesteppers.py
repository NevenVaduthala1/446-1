import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial

# Base Timestepper class
class Timestepper:
    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)

# Explicit Timestepper base class
class ExplicitTimestepper(Timestepper):
    def __init__(self, u, f):
        super().__init__()
        self.u = u
        self.f = f

# Forward Euler method
class ForwardEuler(ExplicitTimestepper):
    def _step(self, dt):
        return self.u + dt * self.f(self.u)

# Lax-Friedrichs method
class LaxFriedrichs(ExplicitTimestepper):
    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt * self.f(self.u)

# Leapfrog method
class Leapfrog(ExplicitTimestepper):
    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt * self.f(self.u)
        else:
            u_temp = self.u_old + 2 * dt * self.f(self.u)
            self.u_old = np.copy(self.u)
            return u_temp

# Multistage (Runge-Kutta)
class Multistage(ExplicitTimestepper):
    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages  # Number of stages (e.g., 3 for RK3)
        self.a = a  # a_ij coefficients (e.g., Butcher tableau)
        self.b = b  # b_i coefficients (e.g., Butcher tableau)

    def _step(self, dt):
        k = [np.zeros_like(self.u) for _ in range(self.stages)]
        k[0] = self.f(self.u)
        for i in range(1, self.stages):
            u_stage = self.u.copy()
            for j in range(i):
                u_stage += dt * self.a[i, j] * k[j]
            k[i] = self.f(u_stage)

        u_new = self.u.copy()
        for i in range(self.stages):
            u_new += dt * self.b[i] * k[i]

        return u_new

# Adams-Bashforth (Multistep)
class AdamsBashforth(ExplicitTimestepper):
    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.dt = dt
        self.f_vals = [f(u)]

    def _compute_coeffs(self):
        if self.steps == 2:
            return [3 / 2, -1 / 2]
        elif self.steps == 3:
            return [23 / 12, -16 / 12, 5 / 12]
        elif self.steps == 4:
            return [55 / 24, -59 / 24, 37 / 24, -9 / 24]
        elif self.steps == 5:
            return [1901 / 720, -1387 / 360, 109 / 30, -637 / 360, 251 / 720]
        elif self.steps == 6:
            return [4277 / 1440, -2641 / 480, 4991 / 720, -3649 / 720, 959 / 480, -95 / 288]
        else:
            raise ValueError("Only steps 2, 3, 4, 5, and 6 are supported.")

    def _step(self, dt):
        if len(self.f_vals) < self.steps:
            new_u = self.u + dt * self.f(self.u)
        else:
            coeffs = self._compute_coeffs()
            new_u = self.u.copy()
            for i, coeff in enumerate(coeffs):
                new_u += dt * coeff * self.f_vals[-(i + 1)]

        self.f_vals.append(self.f(new_u))
        if len(self.f_vals) > self.steps:
            self.f_vals.pop(0)

        return new_u

# Implicit Timestepper for methods like Backward Euler, BDF, Crank-Nicolson
class ImplicitTimestepper(Timestepper):
    def __init__(self, u, L):
        super().__init__()
        self.u = u
        self.L = L  # Differential operator (matrix)
        N = len(u)
        self.I = sparse.eye(N, N)

    def _LUsolve(self, data):
        return self.LU.solve(data)

# Backward Euler method
class BackwardEuler(ImplicitTimestepper):
    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt * self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self._LUsolve(self.u)

# Crank-Nicolson method
class CrankNicolson(ImplicitTimestepper):
    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt / 2 * self.L.matrix
            self.RHS = self.I + dt / 2 * self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self._LUsolve(self.RHS @ self.u)

# Backward Differentiation Formula (BDF) method
class BackwardDifferentiationFormula(ImplicitTimestepper):
    def __init__(self, u, L, steps):
        super().__init__(u, L)
        N = len(u)
        self.I = sparse.eye(N, N)
        self.steps = steps

    def _step(self, dt):
        if self.iter == 0:
            # Initialize for the first iteration
            N = len(self.u)
            global u_vec, A, dt_vec
            u_vec = np.zeros((self.steps + 1, N))
            u_vec[0] = np.copy(self.u)
            A = np.zeros(self.steps + 1)
            dt_vec = np.zeros(self.steps + 1)
            for i in range(self.steps + 1):
                dt_vec[i] = -i * dt
            LHS = self.I - dt * self.L.matrix
            return spla.spsolve(LHS, self.u)

        if self.iter < self.steps and self.iter > 0:
            # Handle first few iterations
            S_len = self.iter + 2
            for i in reversed(range(1, self.iter + 1)):
                u_vec[i] = np.copy(u_vec[i - 1])
            u_vec[0] = np.copy(self.u)
            S = np.zeros((S_len, S_len))
            for i in range(S_len):
                for j in range(S_len):
                    S[i, j] = 1 / factorial(j) * (-i * dt) ** j
            b = [0] * S_len
            b[1] = 1
            a = b @ np.linalg.inv(S)
            A[0:S_len] = a
            LHS = self.L.matrix - A[0] * self.I
            return spla.spsolve(LHS, A[1:self.iter + 2] @ u_vec[0:self.iter + 1])

        # When iter >= self.steps
        b = [0] * (self.steps + 1)
        b[1] = 1
        dt_vec = dt_vec - dt
        for i in reversed(range(1, self.steps + 1)):
            dt_vec[i] = dt_vec[i - 1]
        dt_vec[0] = 0
        S = np.zeros((self.steps + 1, self.steps + 1))
        for i in range(self.steps + 1):
            for j in range(self.steps + 1):
                S[i, j] = 1 / factorial(j) * (dt_vec[i]) ** j
        a = b @ np.linalg.inv(S)
        A = a
        for i in reversed(range(1, self.steps + 1)):
            u_vec[i] = np.copy(u_vec[i - 1])
        u_vec[0] = np.copy(self.u)

        LHS = self.L.matrix - A[0] * self.I
        return spla.spsolve(LHS, A[1:] @ u_vec[:-1])
