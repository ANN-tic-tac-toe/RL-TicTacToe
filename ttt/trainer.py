import numpy as np
import torch
from torch import optim, nn

from ttt.network import DQN
from ttt.player import Player
from ttt.replay_memory import ReplayMemory, Transition
from ttt.tic_env import TictactoeEnv
from ttt.epsilon_strategy import EpsilonStrategy
from ttt.agent import Agent
from policy import Policy
from loguru import logger
import sys
from tqdm import tqdm
from metrics import compute_m_opt, compute_m_rand

from ttt.utils import represent_gird_as_state, from_tuple_to_int

logger.remove()
logger.add(sys.stderr, level="INFO")


class Trainer:
    """
    Trainer of Deep Q-learning Network
    """

    def __init__(self,
                 opponent: Player,
                 policy: Policy,
                 train_strategy: str = "vs-optimal",
                 n_games: int = 20_000,
                 target_update: int = 500,
                 batch_size: int = 64,
                 replay_memory: ReplayMemory = None,
                 gamma: float = 0.99,
                 adam_lr: float = 0.0005,
                 huber_delta: float = 1.,
                 n_actions=9
                 ):
        self.avg_loss_list = []
        self.current_avg_loss_list = []
        self.current_avg_reward_list = []
        self.avg_rewards_list = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = TictactoeEnv()

        self.dqn = DQN(self.device).to(self.device)
        self.target_dqn = DQN(self.device).to(self.device)
        # load the main network weights to the target network
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=adam_lr)
        self.loss = nn.HuberLoss(delta=huber_delta)

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

    def establish_order(self, game_number):
        players_order = None
        if game_number % 2 == 0:
            # I go first
            players_order = [self.agent, self.opponent]
        else:
            # opponent first
            players_order = [self.opponent, self.agent]
        players_order[0].player = "X"
        players_order[1].player = "O"
        return players_order

    # def init_avg_lists(self):
    #     self.avg_rewards_list = []
    #     self.current_avg_reward_list = []
    #     self.avg_loss_list = []
    #     self.current_avg_loss_list = []

    def optimize_model(self):
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(~np.array(batch.is_final))
        state_batch = torch.tensor(batch.state)
        reward_batch = torch.tensor(batch.reward)
        action_batch = torch.tensor(batch.action, device=self.device).reshape(-1, 1)
        state_action_values = self.dqn(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        non_final_next_states = torch.tensor(
            [n_s for n_s, is_f in zip(batch.next_state, batch.is_final) if is_f is False], device=self.device)
        if len(non_final_next_states) != 0:
            next_state_values[non_final_mask] = self.target_dqn(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        criterion = nn.HuberLoss(delta=self.huber_delta)
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradient()
        self.optimizer.step()
        return loss

    def clip_gradient(self):
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)

    def init_replay_memory(self):
        logger.info("Initialize Replay Memory")

        mem_size = 0
        game_number = -1
        self.replay_memory.init_size = 64
        pbar = tqdm(total=self.replay_memory.init_size)
        while mem_size < self.replay_memory.init_size:
            game_number += 1
            self.env.reset()

            players_order = self.establish_order(game_number)
            # loop for a single game, stop when game is finished and add stuff the memory
            for ith_move in range(self.n_actions):
                state_t = represent_gird_as_state(self.env.grid.copy(), my_representation=self.agent.player,
                                                  device=self.device)
                current_player = ith_move % 2
                action_t = players_order[current_player].act(self.env.grid, n=0)

                if self.env.check_valid(action_t):
                    new_gird, is_final, winner = self.env.step(action_t)
                    reward_t = self.env.reward(player=self.agent.player)

                else:
                    new_grid = self.env.grid.copy()  # no change to the grid, because it would mean overwriting a value
                    is_final = True
                    reward_t = -1

                new_state = represent_gird_as_state(new_gird, self.agent.player, self.device)
                # collect the data only for the agent not the optimal player
                if players_order[current_player] is self.agent:
                    self.replay_memory.push(state_t.numpy(), from_tuple_to_int(action_t), reward_t, new_state.numpy(),
                                            is_final)
                    mem_size += 1
                    pbar.update(1)
                if mem_size == self.replay_memory.init_size:
                    break
                if is_final:
                    break
        pbar.close()


    def training_loop(self, optimize=True):
        for ith_game in tqdm(range(self.n_games)):

            self.env.reset()
            state = self.env.grid
            players_order = self.establish_order(ith_game)
            # loop for a single game, stop when game is finished and add stuff the memory
            for ith_move in range(self.n_actions):
                state_t = represent_gird_as_state(self.env.grid.copy(), my_representation=self.agent.player,
                                                  device=self.device)
                current_player = ith_move % 2
                action_t = players_order[current_player].act(self.env.grid, n=ith_game)

                if self.env.check_valid(action_t):
                    new_grid, is_final, winner = self.env.step(action_t)
                    reward_t = self.env.reward(player=self.agent.player)

                else:
                    new_grid = self.env.grid.copy()  # no change to the grid, because it would mean overwriting a value
                    is_final = True
                    reward_t = -1
                new_state = represent_gird_as_state(new_grid, self.agent.player, self.device)

                # collect the data only for the agent not the optimal player
                if players_order[current_player] is self.agent:
                    self.replay_memory.push(state_t.numpy(), from_tuple_to_int(action_t), reward_t, new_state.numpy(),
                                            is_final)
                    self.current_avg_reward_list.append(reward_t)
                    loss = self.optimize_model()
                    self.current_avg_loss_list.append(loss)

                if is_final:
                    break
            if ith_game % self.target_update == 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())
            if ith_game % 250 == 0 and ith_game != 0:
                # save the average reward of the last 250 games
                self.avg_rewards_list.append(sum(self.current_avg_reward_list) / len(self.current_avg_reward_list))
                # reset the list for the new games
                self.current_avg_reward_list = []
                # save the average loss of the last 250 games
                self.avg_loss_list.append(sum(self.current_avg_loss_list) / len(self.current_avg_loss_list))
                current_avg_loss_list = []

                self.m_opt.append(compute_m_opt(self.agent))
                self.m_rand.append(compute_m_rand(self.agent))