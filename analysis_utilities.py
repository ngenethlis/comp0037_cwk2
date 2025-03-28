"""
Utilities for analysis. Originally designed for q1_f.py, but reused in q2_b and q2_g.
"""

import itertools

import numpy as np

from common.airport_map import MapCell
from generalized_policy_iteration.tabular_value_function import \
    TabularValueFunction
from p1.low_level_actions import LowLevelActionType
from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_policy import LowLevelPolicy


def matrix_difference_absolute(matrix_1: np.ndarray, matrix_2: np.ndarray) -> float:
    """
    Gets the difference between two matrices, and then gets the matrix norm of the result.
    Essentially: abs(matrix_1 - matrix_2).

    Also does additional work to turn NaN values into 0, if they exist.
    """
    return np.linalg.norm(np.nan_to_num(matrix_1) - np.nan_to_num(matrix_2))

def matrix_if_differ_difference_absolute(
        matrix_1: np.ndarray, matrix_2: np.ndarray, float_epsilon: float = 10e-8
) -> float:
    """
    Element-wise, outputs a 1 if they differ by at most float_epsilon, 0 if they don't.
    Then, takes the sum of all elements in the matrix.
    """
    return np.sum(
        np.vectorize(lambda x: int(abs(x) >= float_epsilon))(
            (np.nan_to_num(matrix_1) - np.nan_to_num(matrix_2))
        )
    )

def value_function_to_numpy(value_function: TabularValueFunction, width: int,
                            height: int) -> np.ndarray:
    """
    Given a TabluarValueFunction, output an equivalently sized ndarray. Typicaly used as
    a pre-processing step before being passed ot one of the matrix functions
    """
    return np.array(
        [
            [value_function.value(x, y) for x in range(width)]
            for y in range(height)
        ]
    )

def get_optimal_policy( # pylint: disable=too-many-locals
    val_fn: TabularValueFunction,
    environment: LowLevelEnvironment,
    gamma: float = 0.5,
) -> LowLevelPolicy:
    """
    From slide 43, "Extracting the Policy" of the slides "Value Iteration and Efficiency"
    This is required because on-policy _predictor_ doesn\'t update its policy, so we have to infer
    it.

    Returns the policy given a value function and environment.
    """
    policy = LowLevelPolicy("optimal", environment.map())
    width, height = policy.width(), policy.height()

    for x, y in itertools.product(range(width), range(height)):
        best_action = LowLevelActionType.NONE
        best_accum = 0.0

        for action in range(LowLevelActionType.NONE):
            s_primes, rs, ps = environment.next_state_and_reward_distribution(
                (x, y), action
            )

            # type annotation
            s_primes: list[MapCell] = s_primes
            accum = 0.0

            # p(s',r|s,a)[r + gamma V(s')]
            value = sum(
                p * (rs[i] + gamma * val_fn.value(*s_primes[i].coords()))
                for i, p in enumerate(ps)
            )
            accum += value

            if accum > best_accum:
                best_action = action
                best_accum = accum

        # set_action already ignores obstructed & terminal cells, so we
        # don't have to check that here
        policy.set_action(x, y, best_action)

    return policy


def policy_to_numpy(policy: LowLevelPolicy) -> np.ndarray[np.ndarray[int]]:
    """
    Turns a policy into a 2D array of actions
    """
    array = np.zeros((policy.width(), policy.height()))

    for x, y in itertools.product(range(policy.width()), range(policy.height())):
        array[x, y] = policy.greedy_optimal_action(x, y)

    return array

def policy_to_comparable(policy: LowLevelPolicy) -> np.ndarray[np.ndarray[int]]:
    """
    Every cell that we visit no matter the action has the same "cost", so we need
    to figure how much "cost" is incurred until we reach a None
    """
    width, height = policy.width(), policy.height()
    array = np.zeros((width, height))
    policy = policy_to_numpy(policy)

    # for every item in the state space, follow the policy until we reach a terminal
    for x, y in itertools.product(range(width), range(height)):
        accumulator = 0

        inner_x, inner_y = x, y
        visited = {(x, y)}
        action = policy[inner_x, inner_y]

        while action not in [LowLevelActionType.TERMINATE, LowLevelActionType.NONE]:
            accumulator += 1

            match action:
                case LowLevelActionType.MOVE_RIGHT:
                    inner_x += 1
                case LowLevelActionType.MOVE_LEFT:
                    inner_x -= 1
                case LowLevelActionType.MOVE_UP:
                    inner_y += 1
                case LowLevelActionType.MOVE_UP_RIGHT:
                    inner_x += 1
                    inner_y += 1
                case LowLevelActionType.MOVE_UP_LEFT:
                    inner_x -= 1
                    inner_y += 1
                case LowLevelActionType.MOVE_DOWN:
                    inner_y -= 1
                case LowLevelActionType.MOVE_DOWN_LEFT:
                    inner_x -= 1
                    inner_y -= 1
                case LowLevelActionType.MOVE_DOWN_RIGHT:
                    inner_x += 1
                    inner_y -= 1

            if not (0 <= inner_x < width and 0 <= inner_y < height):
                break

            if (inner_x, inner_y) in visited:
                break

            action = policy[inner_x, inner_y]
            visited.add((inner_x, inner_y))

        array[x, y] = accumulator

    return array
