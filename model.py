import numpy as np


class DummyModel:
    def __init__(self, use_sota=False):
        self._step = 0

        if use_sota:
            self._train_loss_mu = np.array([20, 10, 7, 5, 4, 3, 3, 2, 2, 1])
            self._valid_loss_mu = np.array([20, 10, 7, 5, 4, 5, 6, 7, 8, 9])
            self._loss_std = 0.3
        else:
            self._train_loss_mu = np.array([20, 10, 8, 6, 5, 5, 4, 4, 3, 3])
            self._valid_loss_mu = np.array([20, 10, 8, 7, 6, 6, 6, 7, 8, 9])
            self._loss_std = 0.8

    def train(self):
        i = self._step
        loss = self._train_loss_mu[i] + np.random.randn(1) * self._loss_std
        self._step = min(i + 1, len(self._train_loss_mu) - 1)
        return float(loss)

    def valid(self):
        i = self._step
        loss = self._valid_loss_mu[i] + np.random.randn(1) * self._loss_std
        return float(loss)
