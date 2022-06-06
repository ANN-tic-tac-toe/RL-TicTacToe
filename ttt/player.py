import abc
import numpy as np


class Player(abc.ABC):
    """Common interface for a player and agent to chose action based on the current state."""

    def __init__(self, player):
        self.player = player

    def set_player(self, player='X', j=-1):
        self.player = player
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'

    @abc.abstractmethod
    def act(self, state: np.array, **kwargs) -> int:
        """
        Chooses action based on the state.
        Args:
            state: represented by np.array (3,3)
            **kwargs:

        Returns:
            int representing an index in a flatted grid to make the move
        """
        raise NotImplementedError("Implement abstract act method to use this class.")
