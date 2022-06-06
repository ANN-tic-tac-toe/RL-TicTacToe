from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'is_final'))


class ReplayMemory(object):

    def __init__(self, capacity, init_size=None):
        self.memory = deque([], maxlen=capacity)
        self.init_size = init_size if init_size is not None else capacity

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
