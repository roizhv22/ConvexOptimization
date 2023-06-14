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
        max_iter = 10
        epsilon = 1e-3
        alpha = 0.1
        beta = 0.5

        X = X0
        t = 1.0

        for i in range(max_iter):
            print(i)
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
                t = max(t * beta, 2.22e-10)

        return X

    # Construct ai_values from input matrix A
    ai_values = [A[:, i] for i in range(m)]
    X0 = np.eye(n)  # Initial guess for X

    X_optimal = optimize(X0, ai_values)
    return X_optimal


def solveProject(A):
    n = A.shape[0]  # Dimension of X
    m = A.shape[1]  # Number of constraints

    def objective(X):
        return np.log(np.linalg.det(X))

    def constraint(X, ai):
        inv_X = np.linalg.inv(X)
        return np.dot(ai.T, np.dot(inv_X, ai)) - 1

    def project(X, ai_values):
        for ai in ai_values:
            inv_X = np.linalg.inv(X)
            norm_ai = np.linalg.norm(ai)
            if norm_ai > np.sqrt(2):
                ai /= (2 * norm_ai / np.sqrt(2))
            ai_outer = np.outer(ai, ai)
            X = X - np.dot(X, np.dot(ai_outer, X)) / (1 + np.dot(ai, np.dot(inv_X, ai)))
        return X

    # Construct ai_values from input matrix A
    ai_values = [A[:, i] for i in range(m)]
    X = np.eye(n)  # Initial guess for X

    max_iter = 100
    epsilon = 1e-3
    initial_step_size = 0.1  # Initial step size
    reduction_factor = 0.2  # Step size reduction factor

    step_size = initial_step_size

    for i in range(max_iter):
        inv_X = np.linalg.inv(X)
        grad = inv_X - np.sum([np.outer(ai, ai) / np.dot(ai, np.dot(inv_X, ai)) for ai in ai_values], axis=0)
        X_new = X + step_size * grad
        X_new = project(X_new, ai_values)
        if np.linalg.norm(X_new - X) < epsilon:
            break
        X = X_new
        step_size *= reduction_factor  # Reduce the step size

    return X


def score(X, A):
    A_reshaped = np.reshape(A, (-1, A.shape[1], 1))
    scores = np.einsum('...i,ij,...j->...', A_reshaped, X, A_reshaped)
    return np.linalg.det(X), np.mean(
        scores <= 1. + 1e-8)



if __name__ == "__main__":
    started = dt.datetime.now()
    # A = np.load("Examples/gaussian.2.5.npy")
    n, d = 100, 3
    np.random.seed(0)
    A = np.random.randn(n, d) * (np.arange(d) + 1.)
    print(f"started solving at {started}")
    res = solveProject(A)
    # res = solve(A)
    # with open("Results/gaussian.2.5.npy", 'wb') as f:
    #     np.save(f, A)
    print(f"Done, time took:{dt.datetime.now() - started}")
    print(score(res, A))
