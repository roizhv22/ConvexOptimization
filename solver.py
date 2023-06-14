import numpy as np
import datetime as dt


def solveGD(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]  # Dimension of X
    m = A.shape[1]  # Number of constraints

    def objective(X):
        return np.log(np.linalg.det(X))

    def constraint(X, ai):
        inv_X = np.linalg.inv(X)
        return np.dot(ai.T, np.dot(inv_X, ai)) - 1

    def optimize(X0, ai_values):
        max_iter = 100
        epsilon = 1e-3
        alpha = 0.1
        beta = 0.5

        X = X0
        t = 1.0

        for i in range(max_iter):
            inv_X = np.linalg.inv(X)
            grad = inv_X - np.sum(
                [np.outer(ai, ai) / np.dot(ai, np.dot(inv_X, ai)) for ai in
                 ai_values], axis=0)
            norm_grad = np.linalg.norm(grad)

            if norm_grad < epsilon:
                break

            while True:
                X_new = X - t * grad
                if np.all(np.linalg.eigvals(X_new) > 0):
                    obj_new = objective(X_new)
                    con_new = np.max(
                        [constraint(X_new, ai) for ai in ai_values])
                    if obj_new < objective(X) + alpha * t * np.trace(
                            np.dot(grad.T, X - X_new)) and con_new <= 0:
                        X = X_new
                        print("found better objective")
                        break
                t *= beta

        return X

    # Construct ai_values from input matrix A
    ai_values = [A[:, i] for i in range(m)]
    X0 = np.eye(n)  # Initial guess for X

    X_optimal = optimize(X0, ai_values)
    return X_optimal


def solveProjected(A):
    n = A.shape[0]  # Dimension of X
    m = A.shape[1]  # Number of constraints

    def objective(X):
        return np.log(np.linalg.det(X))

    def constraint(X, ai):
        inv_X = np.linalg.inv(X)
        return np.dot(ai.T, np.dot(inv_X, ai)) - 1

    def barrier_term(X, t, ai_values):
        barrier = 0
        for ai in ai_values:
            barrier -= np.log(constraint(X, ai))
        return t * barrier

    def barrier_gradient(X, t, ai_values):
        gradient = np.zeros((n, n))
        for ai in ai_values:
            inv_X = np.linalg.inv(X)
            outer_product = np.outer(ai, ai)
            gradient += inv_X @ outer_product @ inv_X
        return -t * gradient

    # Construct ai_values from input matrix A
    ai_values = [A[:, i] for i in range(m)]
    X = np.eye(n)  # Initial guess for X

    max_iter = 100
    epsilon = 1e-3
    t = 1.0

    for i in range(max_iter):
        objective_value = objective(X) + barrier_term(X, t, ai_values)
        gradient = np.linalg.inv(X) - barrier_gradient(X, t, ai_values)
        norm_gradient = np.linalg.norm(gradient)

        if norm_gradient < epsilon:
            break

        step_size = 0.1 / norm_gradient
        X_new = X - step_size * gradient

        # Project onto the positive definite cone
        eigenvalues, eigenvectors = np.linalg.eig(X_new)
        X_new = eigenvectors @ np.diag(
            np.maximum(eigenvalues, 1e-8)) @ eigenvectors.T

        # Adjust t using a backtracking line search
        while objective(X_new) + barrier_term(X_new, t,
                                              ai_values) > objective_value:
            step_size *= 0.5
            X_new = X - step_size * gradient
            eigenvalues, eigenvectors = np.linalg.eig(X_new)
            X_new = eigenvectors @ np.diag(
                np.maximum(eigenvalues, 1e-8)) @ eigenvectors.T

        X = X_new
        t *= 0.9

    return X


if __name__ == "__main__":
    started = dt.datetime.now()
    A = np.load("Examples/wave.50.4.npy")
    print(f"started solving at {started}")
    res = solveGD(A)
    np.save("results/wave.50.4.npy", A)
    print(f"Done, time took:{dt.datetime.now() - started}")

