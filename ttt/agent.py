from policy import Policy
from utils import represent_gird_as_state, from_int_to_tuple_repr
import torch
from player import Player


class Agent(Player):
    def __init__(self, policy: Policy, player: str, device):
        super().__init__(player)
        self.policy = policy
        self.player = player
        self.device = device

    def act(self, grid, **kwargs):
        state = represent_gird_as_state(grid, my_representation=self.player, device=self.device)
        return from_int_to_tuple_repr(int(self.policy.select_action(state, **kwargs)[0][0]))
