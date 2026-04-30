import numpy as np
from .autograd import Tensor
from .train import evaluate


def load_model_weights(model, path):
    ckpt = np.load(path, allow_pickle=True)
    state = {k: ckpt[k] for k in ckpt.files if k != 'act_name'}
    model.load_state_dict(state)


def confusion_matrix(model, X, y, num_classes=10, batch_size=256):
    from .data import iter_batches
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for xb, yb in iter_batches(X, y, batch_size=batch_size, shuffle=False):
        logits = model(Tensor(xb, requires_grad=False)).data
        pred = np.argmax(logits, axis=1)
        for t, p in zip(yb, pred):
            cm[t, p] += 1
    return cm


def test_model(model, X_test, y_test, num_classes=10):
    model.eval()
    acc = evaluate(model, X_test, y_test)
    cm = confusion_matrix(model, X_test, y_test, num_classes=num_classes)
    print(f'Test Accuracy: {acc:.4f}')
    print('Confusion Matrix (rows=true, cols=pred):')
    print(cm)
    return acc, cm
