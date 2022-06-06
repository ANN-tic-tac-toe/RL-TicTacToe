import numpy as np
import torch
from matplotlib import pyplot as plt
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

    def main_loop(self, calculate_ms=True):
        # it differs from the main phase by optimization and  metrics calculations
        still_mem_init_phase = True
        logger.info("Initialize Replay Memory: Started")
        game_number = -1
        mem_size = 0
        init_reply_memory_pbar = tqdm(total=self.replay_memory.init_size)
        while True:
            self.env.reset()
            game_number += 1
            players_order = self.establish_order(game_number)
            actors_prev_state = None
            actors_prev_action = None
            for ith_move in range(self.n_actions):
                current_player = ith_move % 2
                state_t = represent_gird_as_state(
                    self.env.grid.copy(),
                    my_representation=self.agent.player,
                    device=self.device
                )
                n = 0 if still_mem_init_phase else game_number
                action_t = players_order[current_player].act(
                    self.env.grid,
                    n=n
                )
                if self.env.check_valid(action_t):
                    new_grid, is_final, winner = self.env.step(action_t)
                    reward_t = self.env.reward(player=self.agent.player)
                else:
                    new_grid = self.env.grid.copy()  # no change to the grid, because it would mean overwriting a value
                    is_final = True
                    reward_t = -1
                new_state = represent_gird_as_state(new_grid, self.agent.player, self.device)
                if players_order[current_player] is self.agent:
                    if is_final:
                        # agent won, draw, or made an incorrect move (all of them are final
                        # and are result of the currect move = therefore the currect state
                        # and a new state
                        self.replay_memory.push(
                            state_t.numpy(),
                            from_tuple_to_int(action_t),
                            reward_t,
                            new_state.numpy(),
                            is_final)
                        if still_mem_init_phase:
                            init_reply_memory_pbar.update(1)
                        if not still_mem_init_phase:
                            self.current_avg_reward_list.append(reward_t)
                            loss = self.optimize_model()
                            self.current_avg_loss_list.append(loss.clone().detach().numpy())
                    else:
                        actors_prev_state = state_t
                        actors_prev_action = action_t

                elif players_order[current_player] is self.opponent:
                    if ith_move == 0:
                        # there's nothing to report to the memory for the first move started by opponent
                        continue
                    # it means that the previous move (the actor's move) led to that
                    # it can be (from the point of view of agent)
                    # if it is final then either the agent lost
                    # or Opponent made rule mistake -but the Optimal player doesn't make rule
                    # mistakes
                    # if it's not final then it's a draw but the update is based on exactly the same rule

                    self.replay_memory.push(
                        actors_prev_state.numpy(),
                        from_tuple_to_int(actors_prev_action),
                        reward_t,
                        new_state.numpy(),
                        is_final)
                    if still_mem_init_phase:
                        init_reply_memory_pbar.update(1)
                    if not still_mem_init_phase:
                        self.current_avg_reward_list.append(reward_t)
                        loss = self.optimize_model()
                        self.current_avg_loss_list.append(loss.clone().detach().numpy())
                else:
                    raise ValueError("either agent or opponent")
                if len(self.replay_memory) == self.replay_memory.init_size and still_mem_init_phase:
                    logger.info("Initialize Replay Memory: Done")
                    logger.info("Training Loop: Start")
                    game_number = 0
                    init_reply_memory_pbar.close()
                    still_mem_init_phase = False
                    training_pbar = tqdm(total=self.n_games)
                    break
                if is_final:
                    break

            if not still_mem_init_phase:
                training_pbar.update(1)
                if game_number % self.target_update == 0:
                    self.target_dqn.load_state_dict(self.dqn.state_dict())
                if game_number % 250 == 0 and game_number != 0:
                    # save the average reward of the last 250 games
                    self.avg_rewards_list.append(sum(self.current_avg_reward_list) / len(self.current_avg_reward_list))
                    # reset the list for the new games
                    self.current_avg_reward_list = []
                    # save the average loss of the last 250 games
                    self.avg_loss_list.append(sum(self.current_avg_loss_list) / len(self.current_avg_loss_list))
                    self.current_avg_loss_list = []

                    if calculate_ms:
                        self.m_opt.append(compute_m_opt(self.agent))
                        self.m_rand.append(compute_m_rand(self.agent))

                if game_number == self.n_games:
                    training_pbar.close()
                    logger.info("Training Loop: Done")
                    break

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
