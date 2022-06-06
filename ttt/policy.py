import abc
import random
import torch
import sys

from ttt.epsilon_strategy import EpsilonStrategy


class Policy(abc.ABC):
    @abc.abstractmethod
    def select_action(self, state, **kwargs):
        pass


class EpsilonGreedy(Policy):
    def __init__(self, network, epsilon_strategy: EpsilonStrategy, n_actions: int, device, logger):
        self.network = network
        self.epsilon_strategy = epsilon_strategy
        self.n_actions = n_actions
        self.logger = logger
        self.device = device

    def select_action(self, state, **kwargs):
        self.logger.debug("EpsilonGreedy: selection action")
        sample = random.random()
        if sample >= self.epsilon_strategy.get_epsilon(kwargs["n"]):
            # greedy action
            self.logger.debug("EpsilonGreedy: Greedy Action")
            with torch.no_grad():
                t = self.network(state).max(0)[1].view(1, 1)
                return t
        else:
            # random action
            self.logger.debug("EpsilonGreedy: Random Action")
            with torch.no_grad():
                return torch.tensor(
                    [[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
