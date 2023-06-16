import os
import numpy as np
import cvxpy as cp

EPSILON = np.finfo(float).eps * 1e-6


def objective(X):
    return -np.log(np.linalg.det(X))


def constraint(X, a):
    x_inv = np.linalg.inv(X)
    return np.dot(a, np.dot(x_inv, a.T)) - 1


def project(X):
    U, s, V = np.linalg.svd(X)  # more stable decomp
    s[s < EPSILON] = EPSILON  # enforce projection to PD
    X_proj = np.dot(U, np.dot(np.diag(s), V))
    return X_proj


def Solve(A: np.ndarray) -> np.ndarray:
    X = np.random.randn(A.shape[1], A.shape[1])
    alpha = 0.1
    max_iter = 100000

    for i in range(max_iter):
        grad = - np.linalg.inv(X)
        X -= alpha * grad
        X = (X + X.T) / 2  # enforce symmetry
        X = project(X)
        if np.all(constraint(X, A) <= 0):
            print(f"broken, {i}")
            break

    return np.linalg.inv(X)


def Libsolve(A):
    n, d = A.shape

    # use root instead of covariance matrix
    R = cp.Variable(shape=(d, d), PSD=True)

    # objective and constraints
    obj = cp.Minimize(-cp.log_det(R))
    constraints = [cp.SOC(1., R @ A[i]) for i in range(n)]
    prob = cp.Problem(obj, constraints)

    # solve
    prob.solve(solver=cp.SCS)
    if prob.status != cp.OPTIMAL:
        raise Exception('CVXPY Error')

    # fixing the result and projection
    X = R.value.T @ R.value
    X /= np.max(np.einsum('...i,ij,...j->...', A, X, A))

    return X


# Example usage:
def score(X, A):
    scores = np.einsum('...i,ij,...j->...', A, X, A)
    return np.log(np.linalg.det(X)), np.mean(
        scores <= 1. + 1e-8)  # industrial solvers always miss


if __name__ == "__main__":
    # for filename in os.listdir("Examples"):
    #     if filename.endswith('20.npy'):
    #         print(f"running on {filename}")
    #         A = np.load(os.path.join("Examples", filename))
    #         X = Solve(A)
    #         thierX = Libsolve(A)
    #         print(f" our:{score(X, A)}, lib: {score(thierX, A)}")
    n, d = 100, 3
    np.random.seed(0)
    A = np.random.randn(n, d) * (np.arange(d) + 1.)
    X = Solve(A)
    Xl = Libsolve(A)
    print(score(X, A))
    print(score(Xl, A))
