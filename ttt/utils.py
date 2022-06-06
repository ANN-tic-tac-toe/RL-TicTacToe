import numpy as np
import torch


def from_int_to_tuple_repr(int_rep):
    return int_rep // 3, int_rep % 3


def from_tuple_to_int(tuple_rep):
    return tuple_rep[0] * 3 + tuple_rep[1]


def represent_gird_as_state(grid: np.ndarray, my_representation: str, device):
    """
    Transforms grid to state required by Deep Q-value Network.

    Transforms grid firstly to (2, 3, 3) then flattens it. Needs to know the representation
        in order to correctly represents the state.
    Args:
        grid: (3,3) array with 1 => X, -1 => O
        my_representation: {'X'| 'O'}
        device: pytorch.device


    Returns:
        (18,) state representation
    """
    if grid is None:
        return None
    grid = grid.copy()
    if my_representation == "X":
        my_mapping = 1
        opp_mapping = -1
    elif my_representation == "O":
        my_mapping = -1
        opp_mapping = 1
    else:
        raise ValueError(f"The representation has to be either 'X' or 'O' but is '{my_representation}' instead")
    my_pos = np.array(grid == my_mapping, dtype=int)
    opponent_pos = np.array(grid == opp_mapping, dtype=int)
    state = np.array([my_pos, opponent_pos]).flatten()
    state = torch.tensor(state, dtype=torch.float, device=device)
    return state
