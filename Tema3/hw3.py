import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

train_file = "D:\\CODE\\Facultate\\semestrul 5\\RN\\Tema3\\extended_mnist_train.pkl"
test_file = "D:\\CODE\\Facultate\\semestrul 5\\RN\\Tema3\\extended_mnist_test.pkl"

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

X_all = np.array(train_data) / 255.0
y_all = np.array(train_labels)
X_test = np.array(test_data) / 255.0

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

def accuracy_from_probs(Y_hat, y_true):
    preds = np.argmax(Y_hat, axis=1)
    return np.mean(preds == y_true)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

np.random.seed(42)
n_inputs = 784
n_hidden = 100
n_outputs = 10

epochs = 50
batch_size = 32
learning_rate = 0.002
l2_lambda = 1e-4

val_frac = 0.1
N = X_all.shape[0]
perm0 = np.random.permutation(N)
val_size = int(N * val_frac)
val_idx = perm0[:val_size]
train_idx = perm0[val_size:]

X_train = X_all[train_idx]
y_train = y_all[train_idx]
X_val = X_all[val_idx]
y_val = y_all[val_idx]

W1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2.0 / n_inputs)
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_hidden, n_outputs) * np.sqrt(2.0 / n_hidden)
b2 = np.zeros(n_outputs)

def forward(X, training=True):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    cache = (X, Z1, A1, Z2)
    return Z2, cache

def backward(cache, Y_hat, Y_true, l2_lambda=0.0):
    X, Z1, A1, Z2 = cache
    m = X.shape[0]

    dZ2 = (Y_hat - Y_true) / m
    dW2 = A1.T.dot(dZ2)
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X.T.dot(dZ1)
    db1 = np.sum(dZ1, axis=0)

    if l2_lambda > 0.0:
        dW2 += (l2_lambda / m) * W2
        dW1 += (l2_lambda / m) * W1

    return dW1, db1, dW2, db2

num_batches = int(np.ceil(X_train.shape[0] / batch_size))

for epoch in range(1, epochs + 1):
    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    epoch_loss = 0.0

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        Y_batch = one_hot(y_batch, n_outputs)

        Z2, cache = forward(X_batch, training=True)

        torch_logits = torch.tensor(Z2, dtype=torch.float32)
        torch_labels = torch.tensor(y_batch, dtype=torch.long)
        torch_loss = F.cross_entropy(torch_logits, torch_labels)
        batch_loss = torch_loss.item()
        epoch_loss += batch_loss * X_batch.shape[0]

        Y_hat = torch.softmax(torch_logits, dim=1).detach().numpy()

        dW1, db1, dW2, db2 = backward(cache, Y_hat, Y_batch, l2_lambda=l2_lambda)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    epoch_loss /= X_train.shape[0]

    Z2_train, _ = forward(X_train, training=False)
    logits_train = torch.tensor(Z2_train, dtype=torch.float32)
    loss_train = F.cross_entropy(logits_train, torch.tensor(y_train, dtype=torch.long)).item()
    Y_train_hat = torch.softmax(logits_train, dim=1).detach().numpy()
    acc_train = accuracy_from_probs(Y_train_hat, y_train)

    Z2_val, _ = forward(X_val, training=False)
    logits_val = torch.tensor(Z2_val, dtype=torch.float32)
    loss_val = F.cross_entropy(logits_val, torch.tensor(y_val, dtype=torch.long)).item()
    Y_val_hat = torch.softmax(logits_val, dim=1).detach().numpy()
    acc_val = accuracy_from_probs(Y_val_hat, y_val)

    print(f"Epoch {epoch:02d}/{epochs} | "
          f"TrainLoss {loss_train:.4f} | ValLoss {loss_val:.4f} | "
          f"TrainAcc {acc_train*100:.2f}% | ValAcc {acc_val*100:.2f}%")

Z2_test, _ = forward(X_test, training=False)
logits_test = torch.tensor(Z2_test, dtype=torch.float32)
Y_test_hat = torch.softmax(logits_test, dim=1).detach().numpy()
y_pred_test = np.argmax(Y_test_hat, axis=1)

predictions_csv = {
    "ID": np.arange(len(y_pred_test)),
    "target": y_pred_test
}

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)
print("\nSubmission file saved as 'submission.csv'")
