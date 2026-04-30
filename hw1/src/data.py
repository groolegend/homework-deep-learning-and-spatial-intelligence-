import os
import numpy as np
from PIL import Image


def load_eurosat(root):
    class_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    x_list, y_list = [], []
    for cls in class_names:
        cls_dir = os.path.join(root, cls)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            fpath = os.path.join(cls_dir, fname)
            img = Image.open(fpath).convert('RGB').resize((64, 64))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            x_list.append(arr.reshape(-1))
            y_list.append(class_to_idx[cls])

    X = np.stack(x_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, class_names


def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    np.random.seed(seed)

    n_cls = int(y.max()) + 1
    train_idx, val_idx, test_idx = [], [], []

    for c in range(n_cls):
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


def iter_batches(X, y, batch_size=64, shuffle=True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(X), batch_size):
        b = idx[i:i + batch_size]
        yield X[b], y[b]
