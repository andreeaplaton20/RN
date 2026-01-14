# import numpy as np
#
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
#
# X = np.array([
#     [1, 1, 0, 0, 0],
#     [1, 1, 0, 1, 0],
#     [1, 0, 1, 0, 1],
#     [1, 0, 0, 0, 1],
#     [1, 1, 1, 1, 0],
#     [1, 1, 0, 1, 1],
#     [1, 1, 0, 0, 1],
#     [1, 0, 1, 0, 0],
# ])
#
# y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
#
# w = np.zeros(5)
#
# eta = 0.1
#
# for iteration in range(1000):
#     z = X @ w
#     sigma_z = sigmoid(z)
#     gradient = X.T @ (y - sigma_z)
#     w += eta * gradient
#
# print("Ponderile optime w:", w)
#
# X_test = np.array([
#     [1, 0, 1, 1, 1],  # U
#     [1, 1, 1, 0, 1],  # V
#     [1, 1, 1, 0, 0],  # W
# ])
#
# y_pred = (sigmoid(X_test @ w) >= 0.5).astype(int)
# print("Predicții:", y_pred)
#


import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0]
])
y = np.array([1, 1, 1, 0, 0, 0, 0, 0])

w = np.zeros(5)
max_iter = 100
tol = 1e-4

for i in range(max_iter):
    z = X @ w
    sigma_z = sigmoid(z)

    grad = X.T @ (y - sigma_z)

    S = sigma_z * (1 - sigma_z)
    H = -((X.T * S) @ X)

    delta = np.linalg.solve(H, grad)
    w_new = w - delta

    if np.linalg.norm(w_new - w) < tol:
        w = w_new
        print(f"Convergență după {i + 1} iterații")
        break
    w = w_new
else:
    print("Nu s-a atins convergența în numărul maxim de iterații")

print("Ponderile optime (Newton-Raphson):", w)

X_test = np.array([
    [1, 0, 1, 1, 1],  # U
    [1, 1, 1, 0, 1],  # V
    [1, 1, 1, 0, 0]  # W
])
y_pred = (sigmoid(X_test @ w) >= 0.5).astype(int)
print("Predicții test:", y_pred)

w_gd = np.zeros(5)
eta = 0.1
for i in range(1000):
    z = X @ w_gd
    grad = X.T @ (y - sigmoid(z))
    w_gd += eta * grad
    if np.linalg.norm(grad) < tol:
        print(f"Gradient ascendent a convers după {i + 1} iterații")
        break

print("Ponderile optime (Gradient Ascendent):", w_gd)

