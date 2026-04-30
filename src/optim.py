class SGD:
    def __init__(self, params, lr=0.01, weight_decay=0.0, momentum=0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = float(momentum)
        self._vel = [None for _ in self.params]
        self.nesterov = False

    def step(self):
        for i, p in enumerate(self.params):
            if not p.requires_grad:
                continue
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data
            if self.momentum > 0:
                if self._vel[i] is None:
                    self._vel[i] = grad.copy()
                else:
                    self._vel[i] = self.momentum * self._vel[i] + grad
                if self.nesterov:
                    p.data -= self.lr * (grad + self.momentum * self._vel[i])
                else:
                    p.data -= self.lr * self._vel[i]
            else:
                p.data -= self.lr * grad

    def zero_grad(self):
        for p in self.params:
            if p.requires_grad:
                p.zero_grad()
