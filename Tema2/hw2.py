import pickle
import numpy as np
import pandas as pd


train_file = "D:\\CODE\\Facultate\\semestrul 5\\RN\\Tema2\\extended_mnist_train.pkl"
test_file = "D:\\CODE\\Facultate\\semestrul 5\\RN\\Tema2\\extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)


train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = []
for image, _ in test:
    test_data.append(image.flatten())

X_train = np.array(train_data) / 255.0
y_train = np.array(train_labels)
X_test = np.array(test_data) / 255.0

Y_train = np.eye(10)[y_train]


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(Y, Y_hat):
    return -np.mean(np.sum(Y * np.log(Y_hat + 1e-8), axis=1))


def forward_backward(X, Y, W, b, lr=0.01):
    m = X.shape[0]

    Z = np.dot(X, W) + b
    Y_hat = softmax(Z)

    loss = compute_loss(Y, Y_hat)

    dZ = Y_hat - Y
    dW = np.dot(X.T, dZ) / m
    db = np.sum(dZ, axis=0) / m

    W -= lr * dW
    b -= lr * db

    return W, b, loss, Y_hat


np.random.seed(42)
n_inputs = 784
n_outputs = 10
W = np.random.randn(n_inputs, n_outputs) * 0.01
b = np.zeros(n_outputs)

epochs = 50
batch_size = 128
learning_rate = 0.1

for epoch in range(epochs):
    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    total_loss = 0
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        Y_batch = Y_train[i:i+batch_size]
        W, b, loss, Y_hat = forward_backward(X_batch, Y_batch, W, b, lr=learning_rate)
        total_loss += loss

    print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss:.4f}")


Z_train = np.dot(X_train, W) + b
Y_train_hat = softmax(Z_train)
y_pred_train = np.argmax(Y_train_hat, axis=1)
train_acc = np.mean(y_pred_train == y_train)
print(f"\nTraining accuracy: {train_acc * 100:.2f}%")


Z_test = np.dot(X_test, W) + b
Y_test_hat = softmax(Z_test)
y_pred_test = np.argmax(Y_test_hat, axis=1)


predictions_csv = {
    "ID": np.arange(len(y_pred_test)),
    "target": y_pred_test
}

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)

print("\nSubmission file saved as 'submission.csv'")
