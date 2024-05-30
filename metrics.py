import numpy as np


class Accuracy:
    def forward(self, targets, outputs):
        targets = np.argmax(targets, axis=1)
        outputs = np.argmax(outputs, axis=1)
        acc = np.mean(targets == outputs)
        return acc


class DataLoader:
    def __init__(self, data, targets, batch_size, shuffle=True):
        self.data = np.array(data)
        self.targets = np.array(targets)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loader_data = []
        self.loader_label = []
        self._create_batches()

    def _create_batches(self):
        if self.shuffle:
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)
            self.data = self.data[indices]
            self.targets = self.targets[indices]
        no_of_batches = len(self.data) // self.batch_size
        for i in range(no_of_batches):
            self.loader_data.append(self.data[i * self.batch_size:(i + 1) * self.batch_size,:])
            self.loader_label.append(self.targets[i * self.batch_size:(i + 1) * self.batch_size,:])
        if no_of_batches * self.batch_size < len(self.data):
            self.loader_data.append(self.data[no_of_batches * self.batch_size:])
            self.loader_label.append(self.targets[no_of_batches * self.batch_size:])

    def batch_data(self):
        return self.loader_data, self.loader_label
