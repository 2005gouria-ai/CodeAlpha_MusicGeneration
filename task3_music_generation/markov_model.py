"""
A dependency-light fallback generator: a second-order Markov chain.

It needs no TensorFlow, so you can always produce music even if installing
deep-learning libraries is a problem. The LSTM (model.py) is the primary,
"AI" approach; this is a simple statistical baseline for comparison.
"""

import random
from collections import defaultdict


class MarkovModel:
    def __init__(self, order=2, seed=None):
        self.order = order
        self.transitions = defaultdict(list)
        self.starts = []
        self.rng = random.Random(seed)

    def fit(self, notes):
        for i in range(len(notes) - self.order):
            key = tuple(notes[i:i + self.order])
            self.transitions[key].append(notes[i + self.order])
            if i < max(1, len(notes) // 5):
                self.starts.append(key)
        return self

    def generate(self, length=120):
        if not self.transitions:
            raise SystemExit("Markov model is empty. Add MIDI files and retry.")
        key = self.rng.choice(self.starts)
        output = list(key)
        for _ in range(length):
            nxt = self.transitions.get(key)
            if not nxt:
                key = self.rng.choice(self.starts)
                output.extend(key)
                continue
            choice = self.rng.choice(nxt)
            output.append(choice)
            key = tuple(output[-self.order:])
        return output
