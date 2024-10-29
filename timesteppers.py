import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque

# Base Timestepper class
class Timestepper:
    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.dt = dt
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

class StateVector:
    def __init__(self, variables, axis=0):
        self.axis, self.variables = axis, variables
        self.N = variables[0].shape[axis]
        shape = list(variables[0].shape)
        shape[axis] *= len(variables)
        self.data = np.zeros(tuple(shape))
        self.slices = [(slice(None),) * axis + (slice(i * self.N, (i + 1) * self.N),) for i in range(len(variables))]
        self.gather()

    def gather(self):
        [np.copyto(self.data[s], v) for s, v in zip(self.slices, self.variables)]

    def scatter(self):
        [np.copyto(v, self.data[s]) for s, v in zip(self.slices, self.variables)]

# IMEXTimestepper class
class IMEXTimestepper(Timestepper):
    def __init__(self, eq_set):
        super().__init__()
        self.X, self.M, self.L, self.F = eq_set.X, eq_set.M, eq_set.L, eq_set.F

    def step(self, dt):
        self.X.gather()
        self.X.data = self._step(dt)
        self.X.scatter()
        self.dt, self.t, self.iter = dt, self.t + dt, self.iter + 1

# Euler IMEX scheme
class Euler(IMEXTimestepper):
    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt * self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        RHS = self.M @ self.X.data + dt * self.F(self.X)
        return self.LU.solve(RHS)

# CNAB (Crank-Nicolson Adams-Bashforth) IMEX scheme
class CNAB(IMEXTimestepper):
    def _step(self, dt):
        if self.iter == 0:
            # Euler step
            LHS = self.M + dt * self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt * self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt or self.iter == 1:
                LHS = self.M + dt / 2 * self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5 * dt * self.L @ self.X.data + 3 / 2 * dt * self.FX - 1 / 2 * dt * self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)

# BDFExtrapolate (Backward Differentiation Formula with Extrapolation)
class BDFExtrapolate(IMEXTimestepper):
    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps

    def _step(self, dt):
        if self.iter == 0:
            global X_store, FX_store
            X_store = [np.zeros_like(self.X.data) for _ in range(self.steps + 1)]
            FX_store = [np.zeros_like(self.F(self.X)) for _ in range(self.steps)]
            LHS = self.M + dt * self.L
            FX_store[0], X_store[0] = self.F(self.X), self.X.data.copy()
            return spla.spsolve(LHS, self.M @ self.X.data + dt * FX_store[0])

        if 0 < self.iter < self.steps:
            FX_store[1:self.iter + 1], X_store[1:self.iter + 1] = FX_store[:self.iter], X_store[:self.iter]
            FX_store[0], X_store[0] = self.F(self.X), self.X.data.copy()
            LHS = self.M + dt * self.L
            return spla.spsolve(LHS, self.M @ self.X.data + dt * FX_store[0])

        b_RHS = np.zeros(self.steps + 1)
        b_RHS[1] = 1
        S = np.array([[(1 / factorial(j)) * (-i * dt) ** j for j in range(self.steps + 1)] for i in range(self.steps + 1)])
        a = b_RHS @ np.linalg.inv(S)


        b_RHS2 = np.zeros(self.steps)
        b_RHS2[0] = 1
        S2 = np.array([[(1 / factorial(j)) * (-(i + 1) * dt) ** j for j in range(self.steps)] for i in range(self.steps)])
        b = b_RHS2 @ np.linalg.inv(S2)

        FX_store[1:self.steps], X_store[1:self.steps + 1] = FX_store[:self.steps - 1], X_store[:self.steps]
        FX_store[0], X_store[0] = self.F(self.X), self.X.data.copy()

        LHS = a[0] * self.M + self.L
        RHS = -self.M @ (a[1:] @ X_store[:-1]) + b @ FX_store
        return spla.spsolve(LHS, RHS)
