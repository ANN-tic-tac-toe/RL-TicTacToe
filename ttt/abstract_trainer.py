import abc
import sys

import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from torch import optim, nn
from tqdm import tqdm

from metrics import compute_m_opt, compute_m_rand
from policy import Policy
from ttt.agent import Agent
from ttt.network import DQN
from ttt.player import Player
from ttt.replay_memory import ReplayMemory, Transition
from ttt.tic_env import TictactoeEnv
from ttt.utils import represent_gird_as_state, from_tuple_to_int

logger.remove()
logger.add(sys.stderr, level="INFO")


class AbstractTrainer:
    """
    Trainer of Deep Q-learning Network
    """

    def __init__(self,
                 dqn,
                 opponent: Player,
                 policy: Policy,
                 replay_memory: ReplayMemory = None,
                 n_games: int = 20_000,
                 target_update: int = 500,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 adam_lr: float = 0.0005,
                 huber_delta: float = 1.,
                 n_actions=9):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = TictactoeEnv()

        self.dqn = dqn
        self.target_dqn = DQN(self.device).to(self.device)
        # load the main network weights to the target network
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=adam_lr)
        self.huber_delta = huber_delta
        self.criterion = nn.HuberLoss(delta=self.huber_delta)

        self.replay_memory = replay_memory
        self.opponent = opponent
        self.n_games = n_games
        self.target_update = target_update
        self.batch_size = batch_size
        self.gamma = gamma
        self.agent = Agent(policy, "X", self.device)
        self.n_actions = n_actions
        self.m_rand = []
        self.m_opt = []
        self.avg_loss_list = []
        self.current_avg_loss_list = []
        self.current_avg_reward_list = []
        self.avg_rewards_list = []

    @abc.abstractmethod
    def establish_order(self, game_number):
        raise NotImplementedError("Implement this method to use the abstract class")

    def optimize_model(self):
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.from_numpy(~np.array(batch.is_final))
        state_batch = torch.from_numpy(np.array(batch.state))
        reward_batch = torch.from_numpy(np.array(batch.reward))
        action_batch = torch.tensor(np.array(batch.action), device=self.device).reshape(-1, 1)
        state_action_values = self.dqn(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        non_final_next_states = torch.from_numpy(
            np.array([n_s for n_s, is_f in zip(batch.next_state, batch.is_final) if is_f is False]))
        if len(non_final_next_states) != 0:
            next_state_values[non_final_mask] = self.target_dqn(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradient()
        self.optimizer.step()
        return loss

    def clip_gradient(self):
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)

    @abc.abstractmethod
    def main_loop(self, calculate_ms=True):
       raise NotImplementedError("Implement the method to use this abstract class")

    def plot_avg_loss_reward(self, filename: str = None):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
        f.suptitle("Average reward and loss in time (over 250 samples)", fontsize=16)
        ax1.set_title("Reward")
        ax2.set_title("Loss")
        ax1.set_xlabel("samples (0 => range[0...249])")
        ax2.set_xlabel("samples (0 => range[0...249])")
        ax1.plot(list(range(len(self.avg_rewards_list))), self.avg_rewards_list)
        ax2.plot(list(range(len(self.avg_loss_list))), self.avg_loss_list)
        if filename is not None:
            f.savefig(fname="./../pictures/" + filename, dpi=300)

    def plot_m_metrics(self, filename: str = None):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
        f.suptitle("Metric over random and optimal", fontsize=16)
        ax1.set_title("M rand")
        ax2.set_title("M opt")
        ax1.set_xlabel("samples (0 => range[0...249])")
        ax2.set_xlabel("samples (0 => range[0...249])")
        ax1.plot(list(range(len(self.m_opt))), self.m_opt)
        ax2.plot(list(range(len(self.m_rand))), self.m_rand)
        if filename is not None:
            f.savefig(fname="./../pictures/" + filename, dpi=300)
