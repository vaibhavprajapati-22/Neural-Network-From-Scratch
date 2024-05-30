import numpy as np


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()  # [batch_size, n_inputs]
        self.dinputs[self.inputs <= 0] = 0  # In relu activation function shape of input is same as output


class Tanh:
    def forward(self, inputs):
        self.outputs = np.tanh(inputs)
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = (1 - (self.outputs ** 2)) * dvalues


class Sigmoid:
    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = self.outputs * (1 - self.outputs) * dvalues


class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalue) in enumerate(zip(self.outputs, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)
