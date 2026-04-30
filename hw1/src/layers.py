import numpy as np
from .autograd import Tensor


class Linear:
    def __init__(self, in_dim, out_dim, init='xavier'):
        if init == 'he':
            w = (np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        else:
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            w = np.random.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)
        b = np.zeros((1, out_dim), dtype=np.float32)
        self.W = Tensor(w, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


def relu(x):
    return x.relu()


def tanh(x):
    return x.tanh()


class Dropout:
    def __init__(self, p=0.0):
        self.p = float(p)
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, x):
        if (not self.training) or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = (np.random.rand(*x.data.shape) < keep).astype(np.float32) / keep
        return x * Tensor(mask, requires_grad=False)


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = float(eps)
        self.gamma = Tensor(np.ones((1, dim), dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros((1, dim), dtype=np.float32), requires_grad=True)

    def __call__(self, x):
        mu = x.mean(axis=1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=1, keepdims=True)
        xhat = xc / (var + self.eps).pow(0.5)
        return xhat * self.gamma + self.beta

    def parameters(self):
        return [self.gamma, self.beta]
