# buffer.py
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.max_size = size

    def add(self, s, a, r, ns, d):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d))

    def __len__(self):
        return len(self.buffer)
