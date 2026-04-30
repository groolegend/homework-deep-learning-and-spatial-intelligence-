from .layers import LayerNorm, Linear, Dropout, relu, tanh
import numpy as np

class MLP3:
    def __init__(self, in_dim, h1, h2, out_dim, act='relu', dropout_p=0.0, layernorm=False):
        init = 'he' if act == 'relu' else 'xavier'
        self.fc1 = Linear(in_dim, h1, init=init)
        self.fc2 = Linear(h1, h2, init=init)
        self.fc3 = Linear(h2, out_dim, init='xavier')
        self.act_name = act
        self.act = relu if act == 'relu' else tanh
        self.dropout_p = float(dropout_p)
        self.use_layernorm = bool(layernorm)
        self.ln1 = LayerNorm(h1) if self.use_layernorm else None
        self.ln2 = LayerNorm(h2) if self.use_layernorm else None
        self.drop1 = Dropout(self.dropout_p)
        self.drop2 = Dropout(self.dropout_p)
        self.training = True

    def train(self):
        self.training = True
        self.drop1.train()
        self.drop2.train()

    def eval(self):
        self.training = False
        self.drop1.eval()
        self.drop2.eval()

    def parameters(self):
        params = self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters()
        if self.use_layernorm:
            params += self.ln1.parameters() + self.ln2.parameters()
        return params

    def __call__(self, x):
        h1 = self.act(self.fc1(x))
        if self.use_layernorm:
            h1 = self.ln1(h1)
        h1 = self.drop1(h1)
        h2 = self.act(self.fc2(h1))
        if self.use_layernorm:
            h2 = self.ln2(h2)
        h2 = self.drop2(h2)
        z = self.fc3(h2)
        return z

    def state_dict(self):
        return {
            'fc1.W': self.fc1.W.data,
            'fc1.b': self.fc1.b.data,
            'fc2.W': self.fc2.W.data,
            'fc2.b': self.fc2.b.data,
            'fc3.W': self.fc3.W.data,
            'fc3.b': self.fc3.b.data,
            'act_name': self.act_name,
            'dropout_p': self.dropout_p,
            'use_layernorm': np.array(int(self.use_layernorm), dtype=np.int64),
        }
        if self.use_layernorm:
            sd.update({
                'ln1.gamma': self.ln1.gamma.data,
                'ln1.beta': self.ln1.beta.data,
                'ln2.gamma': self.ln2.gamma.data,
                'ln2.beta': self.ln2.beta.data,
            })
        return sd
    def load_state_dict(self, state):
        self.fc1.W.data = state['fc1.W'].astype('float32')
        self.fc1.b.data = state['fc1.b'].astype('float32')
        self.fc2.W.data = state['fc2.W'].astype('float32')
        self.fc2.b.data = state['fc2.b'].astype('float32')
        self.fc3.W.data = state['fc3.W'].astype('float32')
        self.fc3.b.data = state['fc3.b'].astype('float32')
        if 'dropout_p' in state:
            self.dropout_p = float(state['dropout_p'])
            self.drop1.p = self.dropout_p
            self.drop2.p = self.dropout_p
        if 'use_layernorm' in state:
            self.use_layernorm = bool(int(state['use_layernorm']))
            if self.use_layernorm and self.ln1 is not None and 'ln1.gamma' in state:
                self.ln1.gamma.data = state['ln1.gamma'].astype('float32')
                self.ln1.beta.data = state['ln1.beta'].astype('float32')
                self.ln2.gamma.data = state['ln2.gamma'].astype('float32')
                self.ln2.beta.data = state['ln2.beta'].astype('float32')
