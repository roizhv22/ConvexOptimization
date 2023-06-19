import os
from typing import List

import numpy as np
import cvxpy as cp
import tqdm as tqdm

EPSILON = np.finfo(float).eps * 1e-5
MAX_ITERATION = 20000
K = 500
STEP_SIZE = 0.1


def objective(X: np.ndarray) -> np.ndarray:
    return -np.log(np.linalg.det(X))


def constraint(X: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.einsum('...i,ij,...j->...', a, X, a) - 1


def dijkstra(X: np.ndarray, functionals: List[np.ndarray],
             epsilon=1e-2, max_iter=100) \
        -> np.ndarray:
    """
    implementation of Dijkstra projection algorithm to project into different
    sets
    :param X: input
    :param functionals: functional from the form
    np.diag(np.diag(outer_product)) where a_i is a row in the constraints
    matrix
    :param epsilon: tol
    :param max_iter: num iterations
    :return:
    """
    p = [np.zeros_like(X) for _ in functionals]
    projected_X = project_to_psd(X)
    x_old = projected_X.copy()

    for _ in range(max_iter):
        for i, f in enumerate(functionals):
            y = project_on_constraint(projected_X + p[i], f)
            p[i] = (projected_X + p[i]) - y
            projected_X = y - p[i]

        if np.linalg.norm(projected_X - x_old) < epsilon:
            break

    return project_to_psd(projected_X)


def project_to_psd(X: np.ndarray) -> np.ndarray:
    """
    Perform projection to the PD cone using EVD decomposition
    :param X: a Matrix to project
    :return: projection
    """
    eigvals, eigvecs = np.linalg.eigh(X)
    eigvals = np.maximum(eigvals, EPSILON)

    # Project to PD
    projected_X = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return projected_X


def project_on_constraint(X: np.ndarray, functional: np.ndarray) -> np.ndarray:
    """
    project on the half space defined by each constraints
    :param X: solution
    :param functional: created by the constraint
    :return: projection
    """
    new_X = X

    val = np.sum(functional * new_X)
    if val > 1.0 - EPSILON:
        new_X -= ((val - 1) /
                  np.linalg.norm(functional, ord='fro') ** 2) * functional

    return new_X


def diagonal_outer_product(a: np.ndarray) -> np.ndarray:
    """
    generate the functional according to a constraint vector
    :param a: a vector from the constraint matrix
    :return: matrix represents the functional
    """
    return np.outer(a, a)


def Solve(A: np.ndarray) -> np.ndarray:
    """
    A solver for minimizing the log det problem. solver using
    PGD with Dijkstra algorithm to perform minimization.
    :rtype: constraints
    """
    # generate random PD matrix is a starting step
    arr = np.random.uniform(1, 2, size=(A.shape[1]))
    X = np.diag(arr)

    # generate constraints functionals
    functionals = [diagonal_outer_product(row) for row in A]

    for i in tqdm.tqdm(range(MAX_ITERATION)):
        # GD step
        grad = - np.linalg.pinv(X)
        X -= STEP_SIZE * grad
        new_X = (X + X.T) / 2  # enforce symmetry

        if i % K == 0:
            # perform Dijkstra's projection algorithm each K iterations
            # to improve performances
            new_X = dijkstra(new_X, functionals)

            if np.all(constraint(X, A) <= 0) and (
                    np.linalg.norm(new_X - X) < EPSILON
                    or np.linalg.norm(
                objective(X) - objective(new_X)) < EPSILON):
                break
        else:
            new_X = project_to_psd(X)

        X = new_X

    for f in functionals:
        project_on_constraint(X, f)

    # normalize solution to the primal problem
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
    return np.linalg.inv(X)


def score(X, A):
    scores = np.einsum('...i,ij,...j->...', A, np.linalg.inv(X), A)
    return (np.linalg.det(X)), np.mean(
        scores <= 1. + 1e-8)  # industrial solvers always miss


if __name__ == "__main__":
    for filename in sorted(os.listdir("Examples")):
        if filename.endswith("npy"):
            print(f"running on {filename}")
            A = np.load(os.path.join("Examples", filename))
            X = Solve(A)
            print("our done")
            thierX = Libsolve(A)
            print(f" our:{score(X, A)}, lib: {score(thierX, A)}")
    # n, d = 100, 3
    # np.random.seed(0)
    # A = np.random.randn(n, d) * (np.arange(d) + 1.)
    # X = Solve(A)
    # Xl = Libsolve(A)
    # print(score(X, A))
    # print(score(Xl, A))
