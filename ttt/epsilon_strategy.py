import abc


class EpsilonStrategy(abc.ABC):
    last_n: int
    @abc.abstractmethod
    def get_epsilon(self, n=None):
        raise NotImplementedError("In order to use this class implement abstract method")


class ConstantEpsilon(EpsilonStrategy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def get_epsilon(self, n=None):
        return self.epsilon


class DecreasingEpsilon(EpsilonStrategy):
    def __init__(self, min_epsilon, max_epsilon, n_star):
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.n_star = n_star

    def get_epsilon(self, n=None):
        if n is None:
            n = self.last_n
        else:
            self.last_n = n
        return max(self.min_epsilon, self.max_epsilon * (1 - n / self.n_star))
