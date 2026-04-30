import numpy as np
from .autograd import Tensor


def cross_entropy_with_logits(logits, y, label_smoothing=0.0):
    x = logits.data
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    bsz = x.shape[0]
    ls = float(label_smoothing)
    if ls > 0:
        num_classes = probs.shape[1]
        y_soft = np.full_like(probs, ls / num_classes, dtype=np.float32)
        y_soft[np.arange(bsz), y] += (1.0 - ls)
        loss_val = -(y_soft * np.log(probs + 1e-12)).sum(axis=1).mean().astype(np.float32)
    else:
        loss_val = -np.log(probs[np.arange(bsz), y] + 1e-12).mean().astype(np.float32)
    out = Tensor(np.array(loss_val, dtype=np.float32), requires_grad=logits.requires_grad, _children=(logits,), _op='cross_entropy')

    def _backward():
        if logits.requires_grad:
            grad = probs.copy().astype(np.float32)
            if ls > 0:
                num_classes = probs.shape[1]
                y_soft = np.full_like(grad, ls / num_classes, dtype=np.float32)
                y_soft[np.arange(bsz), y] += (1.0 - ls)
                grad -= y_soft
            else:
                grad[np.arange(bsz), y] -= 1.0
            grad /= bsz
            logits.grad = logits.grad + out.grad * grad

    out._backward = _backward
    return out


def l2_regularization(model, weight_decay):
    if weight_decay <= 0:
        return 0.0
    l2 = 0.0
    for p in model.parameters():
        l2 += float((p.data * p.data).sum())
    return 0.5 * weight_decay * l2
