import numpy as np


def _ensure_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(np.array(x, dtype=np.float32), requires_grad=False)


def _unbroadcast(grad, shape):
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float32) if requires_grad else None
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f'Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})'

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError('grad must be specified for non-scalar tensor.')
            grad = np.ones_like(self.data, dtype=np.float32)

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = self.grad + grad.astype(np.float32)

        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
            _op='add',
        )

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad = other.grad + _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op='neg')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad - out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-_ensure_tensor(other))

    def __rsub__(self, other):
        return _ensure_tensor(other) - self

    def __mul__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
            _op='mul',
        )

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * other.data
                self.grad = self.grad + _unbroadcast(grad_self, self.data.shape)
            if other.requires_grad:
                grad_other = out.grad * self.data
                other.grad = other.grad + _unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = _ensure_tensor(other)
        return self * other.pow(-1.0)

    def __rtruediv__(self, other):
        return _ensure_tensor(other) / self

    def pow(self, power):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, _children=(self,), _op='pow')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad * (power * (self.data ** (power - 1)))

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = _ensure_tensor(other)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            out_data = self.data @ other.data
        out = Tensor(
            out_data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
            _op='matmul',
        )

        def _backward():
            if self.requires_grad:
                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    self.grad = self.grad + out.grad @ other.data.T
            if other.requires_grad:
                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    other.grad = other.grad + self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sum',
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is None:
                    grad = np.ones_like(self.data, dtype=np.float32) * grad
                else:
                    if not keepdims:
                        if isinstance(axis, tuple):
                            for ax in sorted(axis):
                                grad = np.expand_dims(grad, ax)
                        else:
                            grad = np.expand_dims(grad, axis)
                    grad = np.ones_like(self.data, dtype=np.float32) * grad
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        else:
            if isinstance(axis, tuple):
                denom = np.prod([self.data.shape[a] for a in axis])
            else:
                denom = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / float(denom)

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, requires_grad=self.requires_grad, _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad * e

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-12), requires_grad=self.requires_grad, _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad / (self.data + 1e-12)

        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(0.0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='relu')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad * (self.data > 0).astype(np.float32)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op='tanh')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad * (1.0 - t * t)

        out._backward = _backward
        return out
