import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim, nn

from ttt.abstract_trainer import AbstractTrainer
from ttt.network import DQN
from ttt.player import Player
from ttt.replay_memory import ReplayMemory, Transition
from ttt.tic_env import TictactoeEnv
from ttt.agent import Agent
from policy import Policy
from loguru import logger
import sys
from tqdm import tqdm
from metrics import compute_m_opt, compute_m_rand

from ttt.utils import represent_gird_as_state, from_tuple_to_int

logger.remove()
logger.add(sys.stderr, level="INFO")


class SelfPracticeTrainer(AbstractTrainer):
    """
    Trainer of Deep Q-learning Network
    """

    def __init__(self, dqn, opponent: Player, policy: Policy, replay_memory: ReplayMemory = None, n_games: int = 20_000,
                 target_update: int = 500, batch_size: int = 64, gamma: float = 0.99, adam_lr: float = 0.0005,
                 huber_delta: float = 1., n_actions=9):
        super().__init__(dqn, None, policy, replay_memory, n_games, target_update, batch_size, gamma, adam_lr,
                         huber_delta, n_actions)
        # the agent using exactly the same network and so on, but need to be identifiable for the training reasons
        self.second_agent = Agent(policy, "O", self.device)


    def establish_order(self, game_number):
        players_order = None
        if game_number % 2 == 0:
            # I go first
            players_order = [self.agent, self.second_agent]
        else:
            # opponent first
            players_order = [self.second_agent, self.agent]
        players_order[0].player = "X"
        players_order[1].player = "O"
        return players_order

    def main_loop(self, calculate_ms=True):
        still_mem_init_phase = True
        logger.info("Initialize Replay Memory: Started")
        game_number = -1
        init_reply_memory_pbar = tqdm(total=self.replay_memory.init_size)
        while True:
            self.env.reset()
            game_number += 1
            players_order = self.establish_order(game_number)
            actor1_prev_state = None
            actor2_prev_state = None
            actor1_prev_action = None
            actor2_prev_action = None


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
                    reward_t = self.env.reward(player=self.agent.player)###check here too
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

                        # update from the perspective of self.agent
                        self.replay_memory.push(
                            state_t.numpy(),
                            from_tuple_to_int(action_t),
                            reward_t,
                            new_state.numpy(),
                            is_final)
                        #update from the perspective of self.second_agent
                        # (states are saved from the perspective of the self.agent)
                        # here I need to flip them to the perspective of self.second_agent
                        self.replay_memory.push(
                            np.roll(actor2_prev_state.numpy(), 9),
                            from_tuple_to_int(actor2_prev_action),
                            self.env.reward(player=self.second_agent.player),
                            np.roll(new_state.numpy(), 9),
                            is_final)


                        if still_mem_init_phase:
                            init_reply_memory_pbar.update(1)
                        if not still_mem_init_phase:
                            self.current_avg_reward_list.append(reward_t)
                            loss = self.optimize_model()
                            self.current_avg_loss_list.append(loss.clone().detach().numpy())
                            break
                    else:#not final
                        actor1_prev_state = state_t
                        actor1_prev_action = action_t
                        if actor2_prev_action is not None:
                            # update the memory with the stuff
                            self.replay_memory.push(
                                np.roll(actor2_prev_state.numpy(), 9),
                                from_tuple_to_int(actor2_prev_action),
                                self.env.reward(player=self.second_agent.player),
                                np.roll(new_state.numpy(), 9),
                                is_final)
                else:# current is actor2
                    if is_final:
                        self.replay_memory.push(
                            np.roll(state_t.numpy(), 9),
                            from_tuple_to_int(action_t),
                            self.env.reward(player=self.second_agent.player),
                            np.roll(new_state.numpy(), 9),
                            is_final)

                        self.replay_memory.push(
                            actor1_prev_state.numpy(),
                            from_tuple_to_int(actor1_prev_action),
                            self.env.reward(player=self.agent.player),
                            new_state.numpy(),
                            is_final)

                        if still_mem_init_phase:
                            init_reply_memory_pbar.update(1)
                        if not still_mem_init_phase:
                            loss = self.optimize_model()
                            self.current_avg_loss_list.append(loss.clone().detach().numpy())
                    else:#not final
                        actor2_prev_state = state_t
                        actor2_prev_action = action_t
                        if actor1_prev_action is not None:
                            # update the memory with the stuff
                            self.replay_memory.push(
                                actor1_prev_state.numpy(),
                                from_tuple_to_int(actor1_prev_action),
                                self.env.reward(player=self.agent.player),
                                new_state.numpy(),
                                is_final)




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
