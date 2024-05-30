import numpy as np


class Stochastic_Gradient_Descent:
    def __init__(self, lr=1):
        self.lr = lr

    def update_params(self, layer):
        layer.weights -= self.lr * layer.dweights
        if layer.bias is not None:
            layer.bias = self.lr * layer.dbias


class Stochastic_Gradient_Descent_With_Momentum:
    def __init__(self, lr=1, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def initialize_velocities(self, layer):
        self.velocities[layer] = {
            'weights': np.zeros(layer.weights.shape),
            'bias': np.zeros(layer.bias.shape) if layer.bias is not None else None
        }

    def update_params(self, layer):
        if layer not in self.velocities:
            self.initialize_velocities(layer)

        self.velocities[layer]['weights'] = self.momentum * self.velocities[layer]['weights'] + (
                1 - self.momentum) * layer.dweights
        layer.weights -= self.lr * self.velocities[layer]['weights']
        if layer.bias is not None:
            self.velocities[layer]['bias'] = self.momentum * self.velocities[layer]['bias'] + (
                    1 - self.momentum) * layer.dbias
            layer.bias -= self.lr * self.velocities[layer]['bias']


class AdaGrad:
    def __init__(self, lr=0.01, eps=1e-7):
        self.lr = lr
        self.eps = eps
        self.accumulated_grads = {}

    def update_params(self, layer):
        if layer not in self.accumulated_grads:
            self.accumulated_grads[layer] = {
                'dweights': np.zeros_like(layer.weights),
                'dbias': np.zeros_like(layer.bias) if layer.bias is not None else None
            }

        self.accumulated_grads[layer]['dweights'] += layer.dweights ** 2
        if layer.bias is not None:
            self.accumulated_grads[layer]['dbias'] += layer.dbias ** 2

        layer.weights -= self.lr * layer.dweights / (np.sqrt(self.accumulated_grads[layer]['dweights']) + self.eps)
        if layer.bias is not None:
            layer.bias -= self.lr * layer.dbias / (np.sqrt(self.accumulated_grads[layer]['dbias']) + self.eps)


class RMSProp:
    def __init__(self, lr=0.01, gamma=0.9, eps=1e-7):
        self.lr = lr
        self.eps = eps
        self.gamma = gamma
        self.accumulated_grads = {}

    def update_params(self, layer):
        if layer not in self.accumulated_grads:
            self.accumulated_grads[layer] = {
                'dweights': np.zeros_like(layer.weights),
                'dbias': np.zeros_like(layer.bias) if layer.bias is not None else None
            }

        self.accumulated_grads[layer]['dweights'] = self.gamma * self.accumulated_grads[layer]['dweights'] + (
                1 - self.gamma) * (layer.dweights ** 2)
        if layer.bias is not None:
            self.accumulated_grads[layer]['dbias'] = self.gamma * self.accumulated_grads[layer]['dbias'] + (
                    1 - self.gamma) * (layer.dbias ** 2)

        layer.weights -= self.lr * layer.dweights / (np.sqrt(self.accumulated_grads[layer]['dweights']) + self.eps)
        if layer.bias is not None:
            layer.bias -= self.lr * layer.dbias / (np.sqrt(self.accumulated_grads[layer]['dbias']) + self.eps)


class Adadelta:
    def __init__(self, gamma=0.9, eps=1e-7):
        self.eps = eps
        self.gamma = gamma
        self.accumulated_grads = {}
        self.accumulated_updates = {}

    def update_params(self, layer):
        if layer not in self.accumulated_grads:
            self.accumulated_grads[layer] = {
                'dweights': np.zeros_like(layer.weights),
                'dbias': np.zeros_like(layer.bias) if layer.bias is not None else None
            }

        if layer not in self.accumulated_updates:
            self.accumulated_updates[layer] = {
                'dweights': np.zeros_like(layer.weights),
                'dbias': np.zeros_like(layer.bias) if layer.bias is not None else None
            }

        self.accumulated_grads[layer]['dweights'] = self.gamma * self.accumulated_grads[layer]['dweights'] + (
                1 - self.gamma) * (layer.dweights ** 2)
        if layer.bias is not None:
            self.accumulated_grads[layer]['dbias'] = self.gamma * self.accumulated_grads[layer]['dbias'] + (
                    1 - self.gamma) * (layer.dbias ** 2)

        update_param_weights = - (np.sqrt(self.accumulated_updates[layer]['dweights'] + self.eps) * layer.dweights / (
                np.sqrt(self.accumulated_grads[layer]['dweights']) + self.eps))
        layer.weights += update_param_weights
        self.accumulated_updates[layer]['dweights'] = self.gamma * self.accumulated_updates[layer]['dweights'] + (
                1 - self.gamma) * (update_param_weights ** 2)

        if layer.bias is not None:
            update_param_bias = - (np.sqrt(self.accumulated_updates[layer]['dbias'] + self.eps) * layer.dbias / (
                    np.sqrt(self.accumulated_grads[layer]['dbias']) + self.eps))
            layer.bias += update_param_bias
            self.accumulated_updates[layer]['dbias'] = self.gamma * self.accumulated_updates[layer]['dbias'] + (
                    1 - self.gamma) * (update_param_bias ** 2)


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.99, eps=1e-7):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.itr = 0

    def update_params(self, layer):
        if not hasattr(layer, 'Vdw'):
            layer.Vdw = np.zeros_like(layer.weights)
            layer.Sdw = np.zeros_like(layer.weights)
        if layer.bias is not None and not hasattr(layer, 'Vdb'):
            layer.Vdb = np.zeros_like(layer.bias)
            layer.Sdb = np.zeros_like(layer.bias)

        self.itr += 1

        layer.Vdw = self.beta1 * layer.Vdw + (1 - self.beta1) * layer.dweights
        vdw_corr = layer.Vdw / (1 - (self.beta1 ** self.itr))
        layer.Sdw = self.beta2 * layer.Sdw + (1 - self.beta2) * (layer.dweights ** 2)
        sdw_corr = layer.Sdw / (1 - (self.beta2 ** self.itr))
        layer.weights -= self.lr * vdw_corr / (np.sqrt(sdw_corr) + self.eps)

        if layer.bias is not None:
            layer.Vdb = self.beta1 * layer.Vdb + (1 - self.beta1) * layer.dbias
            vdb_corr = layer.Vdb / (1 - (self.beta1 ** self.itr))
            layer.Sdb = self.beta2 * layer.Sdb + (1 - self.beta2) * (layer.dbias ** 2)
            sdb_corr = layer.Sdb / (1 - (self.beta2 ** self.itr))
            layer.bias -= self.lr * vdb_corr / (np.sqrt(sdb_corr) + self.eps)


class Learning_Rate_Scheduler:
    def __init__(self, learning_rate, decay_rate, decay_step):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.curr_learning_rate = learning_rate
        self.step = 0
        self.decay_step = decay_step

    def update_lr(self):
        if (self.step + 1) % self.decay_step == 0:
            self.curr_learning_rate = self.curr_learning_rate * self.decay_rate
        return self.curr_learning_rate

    def update_step(self):
        self.step += 1
