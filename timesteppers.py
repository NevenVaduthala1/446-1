import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque

class Timestepper:
    def __init__(self):
        self.t = self.iter = 0
        self.dt = None

    def step(self, dt):
        self.X.gather()
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t, self.iter, self.dt = self.t + dt, self.iter + 1, dt

    def evolve(self, dt, time):
        while self.t < time - 1e-8: self.step(dt)


class ExplicitTimestepper(Timestepper):
    def __init__(self, eq_set):
        super().__init__()
        self.X, self.F, self.BC = eq_set.X, eq_set.F, getattr(eq_set, 'BC', None)

    def step(self, dt):
        super().step(dt)
        if self.BC:
            self.BC(self.X)
            self.X.scatter()

class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.X.data + dt*self.F(self.X)

class LaxFriedrichs(ExplicitTimestepper):
    def __init__(self, eq_set):
        super().__init__(eq_set)
        N = len(self.X.data)
        self.A = sparse.diags([1/2, 1/2], [-1, 1], shape=(N, N)).tolil()
        self.A[0, -1] = self.A[-1, 0] = 1/2
        self.A = self.A.tocsr()

    def _step(self, dt):
        return self.A @ self.X.data + dt * self.F(self.X)


class Leapfrog(ExplicitTimestepper):
    def _step(self, dt):
        if self.iter == 0:
            self.X_old = self.X.data.copy()
            return self.X.data + dt * self.F(self.X)
        self.X_old, X_temp = self.X.data.copy(), self.X_old + 2 * dt * self.F(self.X)
        return X_temp

class Multistage(ExplicitTimestepper):
    def __init__(self, eq_set, stages, a, b):
        super().__init__(eq_set)
        self.stages, self.a, self.b = stages, a, b
        self.X_list = [StateVector([np.copy(var) for var in self.X.variables]) for _ in range(stages)]
        self.K_list = [self.X.data.copy() for _ in range(stages)]

    def _step(self, dt):
        np.copyto(self.X_list[0].data, self.X.data)
        for i in range(1, self.stages):
            self.K_list[i-1] = self.F(self.X_list[i-1])
            np.copyto(self.X_list[i].data, self.X.data)
            self.X_list[i].data += sum(self.a[i, j] * dt * self.K_list[j] for j in range(i))
            if self.BC:
                self.BC(self.X_list[i])

        self.K_list[-1] = self.F(self.X_list[-1])
        self.X.data += sum(self.b[i] * dt * self.K_list[i] for i in range(self.stages))
        return self.X.data


def RK22(eq_set):
    return Multistage(eq_set, 2, np.array([[0, 0], [0.5, 0]]), np.array([0, 1]))


class AdamsBashforth(ExplicitTimestepper):
    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        self.f_list = deque(np.copy(self.X.data) for _ in range(steps))

    def _step(self, dt):
        self.f_list.rotate(1)
        self.f_list[0] = self.F(self.X)
        coeffs = self._coeffs(min(self.iter + 1, self.steps))
        self.X.data += dt * sum(c * f for c, f in zip(coeffs, self.f_list))
        return self.X.data

    def _coeffs(self, num):
        j = np.arange(1, num + 1)[:, None]
        return np.linalg.solve((-j.T)**(j-1) / factorial(j-1), (-1)**(j+1) / factorial(j))


class ImplicitTimestepper(Timestepper):
    def __init__(self, eq_set, axis):
        super().__init__()
        self.axis = axis
        self.X, self.M, self.L = eq_set.X, eq_set.M, eq_set.L
        self.I = sparse.eye(len(self.X.data))

    def _LUsolve(self, data):
        return self.LU.solve(data if self.axis == 0 else data.T).T if self.axis else self.LU.solve(data)


class BackwardEuler(ImplicitTimestepper):
    def _step(self, dt):
        if dt != self.dt:
            self.LU = spla.splu((self.M + dt * self.L).tocsc(), permc_spec='NATURAL')
        return self._LUsolve(self.X.data)


class CrankNicolson(ImplicitTimestepper):
    def _step(self, dt):
        if dt != self.dt:
            self.LU, self.RHS = spla.splu((self.M + dt / 2 * self.L).tocsc()), self.M - dt / 2 * self.L
        return self._LUsolve(apply_matrix(self.RHS, self.X.data, self.axis))


class BackwardDifferentiationFormula(ImplicitTimestepper):
    def __init__(self, u, L, steps):
        super().__init__(u, L)
        self.I = sparse.eye(len(u))
        self.steps = steps
        self.u_vec = np.zeros((steps + 1, len(u)))
        self.A = np.zeros(steps + 1)
        self.dt_vec = np.zeros(steps + 1)

    def _step(self, dt):
        if self.iter == 0:
            self.u_vec[0] = self.u
            self.dt_vec = -dt * np.arange(self.steps + 1)
            return spla.spsolve(self.I - dt * self.L, self.u)

        if self.iter < self.steps:
            self.u_vec[1:self.iter + 1] = self.u_vec[:self.iter]
            self.u_vec[0] = self.u
            S = [[(-i * dt) ** j / factorial(j) for j in range(self.iter + 2)] for i in range(self.iter + 2)]
            self.A[:self.iter + 2] = np.linalg.solve(S, [0, 1] + [0] * self.iter)
            LHS = self.L - self.A[0] * self.I
            return spla.spsolve(LHS, self.A[1:self.iter + 2] @ self.u_vec[:self.iter + 1])

        self.dt_vec = np.roll(self.dt_vec - dt, 1)
        S = [[self.dt_vec[i] ** j / factorial(j) for j in range(self.steps + 1)] for i in range(self.steps + 1)]
        self.A = np.linalg.solve(S, [0, 1] + [0] * (self.steps - 1))
        self.u_vec[1:], self.u_vec[0] = self.u_vec[:-1], self.u
        return spla.spsolve(self.L - self.A[0] * self.I, self.A[1:] @ self.u_vec[:-1])


class StateVector:
    def __init__(self, variables, axis=0):
        self.axis, self.variables = axis, variables
        shape = list(variables[0].shape)
        shape[axis] *= len(variables)
        self.data = np.zeros(tuple(shape))
        self.N = variables[0].shape[axis]
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[(slice(None),) * self.axis + (slice(i * self.N, (i + 1) * self.N),)], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[(slice(None),) * self.axis + (slice(i * self.N, (i + 1) * self.N),)])



class IMEXTimestepper(Timestepper):
    def __init__(self, eq_set):
        super().__init__()
        self.X, self.M, self.L, self.F = eq_set.X, eq_set.M, eq_set.L, eq_set.F

    def step(self, dt):
        self.X.gather(), self.X.scatter()
        self.X.data = self._step(dt)
        self.t, self.dt, self.iter = self.t + dt, dt, self.iter + 1


class Euler(IMEXTimestepper):
    def _step(self, dt):
        if dt != self.dt:
            self.LU = spla.splu((self.M + dt * self.L).tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.M @ self.X.data + dt * self.F(self.X))



class CNAB(IMEXTimestepper):
    def _step(self, dt):
        if self.iter == 0 or dt != self.dt or self.iter == 1:
            self.LU = spla.splu((self.M + (dt if self.iter == 0 else dt / 2) * self.L).tocsc())
        self.FX, self.dt = self.F(self.X), dt
        RHS = self.M @ self.X.data + (dt if self.iter == 0 else -0.5 * dt * self.L @ self.X.data + 1.5 * dt * self.FX - 0.5 * dt * self.FX_old)
        self.FX_old = self.FX
        return self.LU.solve(RHS)


class BDFExtrapolate(IMEXTimestepper):
    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        self.X_store = [self.X.data.copy()] * (steps + 1)
        self.FX_store = [self.F(self.X)] * steps

    def _step(self, dt):
        if self.iter == 0:
            LHS = self.M + dt * self.L
            return spla.spsolve(LHS, self.M @ self.X.data + dt * self.F(self.X))
        if self.iter < self.steps:
            self.X_store = [self.X.data.copy()] + self.X_store[:-1]
            self.FX_store = [self.F(self.X)] + self.FX_store[:-1]
            LHS = self.M + dt * self.L
            return spla.spsolve(LHS, self.M @ self.X.data + dt * self.FX_store[0])

        # Compute coefficients
        dt_powers = [(-i * dt) ** np.arange(self.steps + 1) / factorial(np.arange(self.steps + 1)) for i in range(self.steps + 1)]
        a = np.linalg.solve(dt_powers, [0, 1] + [0] * (self.steps - 1))
        dt_powers_extrap = [(-(i + 1) * dt) ** np.arange(self.steps) / factorial(np.arange(self.steps)) for i in range(self.steps)]
        b = np.linalg.solve(dt_powers_extrap, [1] + [0] * (self.steps - 1))

        self.X_store = [self.X.data.copy()] + self.X_store[:-1]
        self.FX_store = [self.F(self.X)] + self.FX_store[:-1]
        LHS = a[0] * self.M + self.L
        RHS = -self.M @ (a[1:] @ self.X_store[1:]) + b @ self.FX_store
        return spla.spsolve(LHS, RHS)

