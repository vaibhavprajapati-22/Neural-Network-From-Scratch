import numpy as np

np.random.seed(42)


class Linear_Layer:
    def __init__(self, n_inputs, n_outputs, bias_=True):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = 0.1 * np.random.randn(n_inputs, n_outputs)  # wights shape ->[n_inputs, n_outputs]
        if bias_:
            self.bias = np.zeros((1, n_outputs))  # bias shape ->[1, n_outputs]
        else:
            self.bias = None

    def forward(self, inputs):  # output shape -> [batch_size, n_outputs]
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias if self.bias is not None else np.dot(inputs, self.weights)

    def num_parameters(self):
        return self.n_inputs * self.n_outputs + self.n_outputs * 1 if self.bias is not None else self.n_inputs * self.n_outputs

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T,
                               dvalues)  # [n_inputs, batch_size] * [batch_size, n_outputs] -> [n_inputs, n_outputs]
        if self.bias is not None:
            self.dbias = np.sum(dvalues, axis=0, keepdims=True)  # [batch_size, n_outputs] -> [1, n_outputs]
        self.dinputs = np.dot(dvalues,
                              self.weights.T)  # [batch_size, n_outputs] * [n_outputs, n_inputs] -> [batch_size, n_inputs]
