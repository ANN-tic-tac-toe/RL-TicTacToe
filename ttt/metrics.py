from ttt.optimal_player import OptimalPlayer
from ttt.tic_env import TictactoeEnv
from ttt.utils import represent_gird_as_state


def compute_m_opt(my_agent, N=500, max_moves=9):
    opt = OptimalPlayer(epsilon=0)
    return compute_m_metric(my_agent, opt, N, max_moves)


def compute_m_rand(my_agent, N=500, max_moves=9):
    rand = OptimalPlayer(epsilon=1)
    return compute_m_metric(my_agent, rand, N, max_moves)


def compute_m_metric(my_agent, comparison_policy, N=500, max_moves=9):
    env = TictactoeEnv()
    n_wins = 0
    n_loss = 0
    for nth_game in range(N):
        env.reset()
        players_order = None
        if nth_game < 250:
            # I go first
            players_order = [my_agent, comparison_policy]
        else:
            # opponent first
            players_order = [comparison_policy, my_agent]

        players_order[0].player = "X"
        players_order[1].player = "O"
        # loop through a single game, stop when game is finished either by someone win/draw-at-the-end/unavailable-move
        for ith_move in range(max_moves):
            current_player = ith_move % 2
            action_t = players_order[current_player].act(env.grid, n=None)
            if env.check_valid(action_t):
                new_gird, is_end, winner = env.step(action_t)
            else:
                winner = players_order[(current_player + 1) % 2].player
                is_end = True

            if is_end:
                # The game has ended
                break

        if winner == my_agent.player:
            n_wins += 1
        elif winner == comparison_policy.player:
            n_loss += 1
        else:
            # draw
            continue
    return (n_wins - n_loss)/N
