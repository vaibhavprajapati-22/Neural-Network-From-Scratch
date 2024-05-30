import numpy as np


class Categorical_Cross_Entropy_Loss:
    def forward(self, targets, outputs):
        outputs = np.clip(outputs, 1e-7, 1 - 1e-7)
        probs = np.sum(targets * outputs, axis=1)
        outputs = np.mean(-np.log(probs))
        return outputs

    def backward(self, targets, outputs):  # targets are the ground truths and outputs are the predicted values
        self.dinputs = - targets / outputs
        self.dinputs /= targets.shape[0]


class Binary_Cross_Entropy_Loss:
    def forward(self, targets, outputs):
        outputs = np.clip(outputs, 1e-7, 1 - 1e-7)
        loss = -(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))
        loss = np.mean(loss, axis=-1)
        return loss

    def backward(self, targets, outputs):  # outputs and targets shape should be [batch_size, 1]
        outputs = np.clip(outputs, 1e-7, 1 - 1e-7)
        self.dinputs = -(targets / outputs - (1 - targets) / (1 - outputs))
        self.dinputs /= targets.shape[0]
