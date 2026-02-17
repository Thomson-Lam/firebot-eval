import random

import numpy as np
import torch


class RunningNormalizer:
    """Online mean/std normalizer using Welford's algorithm."""

    def __init__(self, shape: int, clip: float = 10.0, alpha: float = 0.999):
        self.shape = shape
        self.clip = clip
        self.alpha = alpha
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
        else:
            # EMA-style update for stability during co-training
            self.mean = self.alpha * self.mean + (1 - self.alpha) * batch_mean
            self.var = self.alpha * self.var + (1 - self.alpha) * batch_var
            self.count += batch_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + 1e-8)
        normed = (x - self.mean) / std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
