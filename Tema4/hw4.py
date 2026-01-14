import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = "/kaggle/input/fii-nn-2025-homework-4/extended_mnist_train.pkl"
test_path = "/kaggle/input/fii-nn-2025-homework-4/extended_mnist_test.pkl"

train_raw = pd.read_pickle(train_path)
test_raw = pd.read_pickle(test_path)


def process_data(raw_data):
    try:
        if isinstance(raw_data, (list, tuple)) and len(raw_data) > 0 and isinstance(raw_data[0], (list, tuple)):
            images, labels = zip(*raw_data)
            return np.stack(images), np.array(labels)
        elif isinstance(raw_data, tuple) and len(raw_data) == 2:
            return raw_data[0], raw_data[1]
    except:
        pass
    return np.stack(raw_data), None


X_train, y_train = process_data(train_raw)
X_test, _ = process_data(test_raw)

X_train = np.array(X_train, dtype=np.float32)
if y_train is not None:
    y_train = np.array(y_train, dtype=np.int64)
X_test = np.array(X_test, dtype=np.float32)

num_classes = 10

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


class FastMNIST(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.size == 784:
            img = img.reshape(28, 28)
        img = img.astype(np.uint8)

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            return img, self.labels[idx]
        return img, 0


batch_size = 4096
num_workers = 2

train_ds = FastMNIST(X_train, y_train, transform=train_transforms)
test_ds = FastMNIST(X_test, labels=None, transform=test_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True, persistent_workers=True)


class WideMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(WideMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model = WideMLP(784, num_classes).to(device)

num_epochs = 22
learning_rate = 0.008

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                          steps_per_epoch=len(train_loader),
                                          epochs=num_epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()

    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            scores = model(x)
            loss = criterion(scores, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    print(f"Ep {epoch + 1}/{num_epochs} Done. (LR: {scheduler.get_last_lr()[0]:.5f})")

model.eval()
all_preds = []

with torch.no_grad():
    for x, _ in tqdm(test_loader):
        x = x.to(device)
        with torch.cuda.amp.autocast():
            scores = model(x)
            preds = scores.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

df = pd.DataFrame({"ID": range(0, len(all_preds)), "target": all_preds})
df.to_csv("submission.csv", index=False)