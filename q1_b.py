#!/usr/bin/env python3

"""
Created on 7 Mar 2023

@author: steam
"""

import multiprocessing as mp
import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple

from common.scenarios import test_three_row_scenario
from common.airport_map import AirportMap, MapCell
from common.airport_map_drawer import AirportMapDrawer

from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor
from monte_carlo.monte_carlo_policy_predictor import MonteCarloPolicyPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator
from generalized_policy_iteration.tabular_value_function import TabularValueFunction

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer
from p1.low_level_policy import LowLevelPolicy


class WorkReturnValue(NamedTuple):
    on_value: float
    off_value: float
    on_policy: float
    off_policy: float
    first_visit: bool
    episode_count: int


# From slide 43, "Extracting the Policy" of the slidess "Value Iteration and Efficiency"
# This is required because on-policy _predictor_ doesn't update its policy, so we have to infer
# it.
def get_optimal_policy(
    val_fn: TabularValueFunction,
    environment: LowLevelEnvironment,
    gamma: float = 0.5,
) -> LowLevelPolicy:
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


def policy_to_numpy(policy: LowLevelPolicy) -> np.ndarray[int]:
    array = np.zeros((policy.width(), policy.height()))

    for x, y in itertools.product(range(policy.width()), range(policy.height())):
        array[x, y] = policy.action(x, y)

    return array


def matrix_difference_absolute(matrix_1: np.ndarray, matrix_2: np.ndarray) -> float:
    return np.linalg.norm(np.nan_to_num(matrix_1) - np.nan_to_num(matrix_2))


def work(args: tuple[bool, int]) -> WorkReturnValue:
    first_visit, episode_count = args
    airport_map, drawer_height = test_three_row_scenario()
    env = LowLevelEnvironment(airport_map)
    env.set_nominal_direction_probability(0.8)

    # Policy to evaluate
    pi = env.initial_policy()
    pi.set_epsilon(0)
    pi.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    pi.set_action(14, 2, LowLevelActionType.MOVE_DOWN)

    # Policy evaluation algorithm
    pe = PolicyEvaluator(env)
    pe.set_policy(pi)
    # v_pe = ValueFunctionDrawer(pe.value_function(), drawer_height)
    pe.evaluate()
    # v_pe.update()
    # v_pe.update()

    # On policy MC predictor
    mcpp = OnPolicyMCPredictor(env)
    mcpp.set_target_policy(pi)
    mcpp.set_experience_replay_buffer_size(64)

    # Q1b: Experiment with this value
    mcpp.set_use_first_visit(first_visit)

    # v_mcpp = ValueFunctionDrawer(mcpp.value_function(), drawer_height)

    # Off policy MC predictor
    mcop = OffPolicyMCPredictor(env)
    mcop.set_target_policy(pi)
    mcop.set_experience_replay_buffer_size(64)
    b = env.initial_policy()
    b.set_epsilon(0.2)
    mcop.set_behaviour_policy(b)

    # Q1b: Experiment with this value
    mcop.set_use_first_visit(first_visit)

    # v_mcop = ValueFunctionDrawer(mcop.value_function(), drawer_height)

    for _ in range(episode_count):
        mcpp.evaluate()
        # v_mcpp.update()
        mcop.evaluate()
        # v_mcop.update()

    # Sample way to generate outputs
    # v_pe.save_screenshot("q1_b_truth_pe.pdf")
    # v_mcop.save_screenshot("q1_b_mc-off_pe.pdf")
    # v_mcpp.save_screenshot("q1_b_mc-on_pe.pdf")

    on_policy = policy_to_numpy(get_optimal_policy(mcop.value_function(), env))
    off_policy = policy_to_numpy(get_optimal_policy(mcpp.value_function(), env))
    ideal_policy = policy_to_numpy(get_optimal_policy(pe.value_function(), env))

    mcpp_values = np.array(
        [
            [mcpp.value_function().value(x, y) for x in range(env.map().width())]
            for y in range(env.map().height())
        ]
    )
    mcop_values = np.array(
        [
            [mcop.value_function().value(x, y) for x in range(env.map().width())]
            for y in range(env.map().height())
        ]
    )
    pe_values = np.array(
        [
            [pe.value_function().value(x, y) for x in range(env.map().width())]
            for y in range(env.map().height())
        ]
    )
    return WorkReturnValue(
        matrix_difference_absolute(mcpp_values, pe_values),
        matrix_difference_absolute(mcop_values, pe_values),
        matrix_difference_absolute(on_policy, ideal_policy),
        matrix_difference_absolute(off_policy, ideal_policy),
        first_visit,
        episode_count,
    )


if __name__ == "__main__":
    MAX_EPISODE_COUNT = 100
    with mp.Pool() as pool:
        results = pool.map(
            work,
            itertools.product([True, False], list(range(0, MAX_EPISODE_COUNT + 1))),
        )

    ys_on_fv = [
        r[0]
        for r in sorted(
            [
                (result.on_value, result.episode_count)
                for result in results
                if result.first_visit
            ],
            key=lambda t: t[1],
        )
    ]
    ys_off_fv = [
        r[0]
        for r in sorted(
            [
                (result.off_value, result.episode_count)
                for result in results
                if result.first_visit
            ],
            key=lambda t: t[1],
        )
    ]
    ys_on_mv = [
        r[0]
        for r in sorted(
            [
                (result.on_value, result.episode_count)
                for result in results
                if not result.first_visit
            ],
            key=lambda t: t[1],
        )
    ]
    ys_off_mv = [
        r[0]
        for r in sorted(
            [
                (result.off_value, result.episode_count)
                for result in results
                if not result.first_visit
            ],
            key=lambda t: t[1],
        )
    ]

    ys_on_pol_fv = [
        r[0]
        for r in sorted(
            [
                (result.on_policy, result.episode_count)
                for result in results
                if result.first_visit
            ],
            key=lambda t: t[1],
        )
    ]
    ys_off_pol_fv = [
        r[0]
        for r in sorted(
            [
                (result.off_policy, result.episode_count)
                for result in results
                if result.first_visit
            ],
            key=lambda t: t[1],
        )
    ]
    ys_on_pol_mv = [
        r[0]
        for r in sorted(
            [
                (result.on_policy, result.episode_count)
                for result in results
                if not result.first_visit
            ],
            key=lambda t: t[1],
        )
    ]
    ys_off_pol_mv = [
        r[0]
        for r in sorted(
            [
                (result.off_policy, result.episode_count)
                for result in results
                if not result.first_visit
            ],
            key=lambda t: t[1],
        )
    ]

    # Value "Difference"
    xs = list(range(0, MAX_EPISODE_COUNT + 1))
    plt.plot(xs, ys_on_fv, color="red")
    plt.plot(xs, ys_on_mv, color="purple")

    plt.plot(xs, ys_off_fv, color="blue")
    plt.plot(xs, ys_off_mv, color="green")
    plt.title("Value Difference")
    plt.show()

    # Policy "Difference"
    plt.plot(xs, ys_on_pol_fv, color="red")
    plt.plot(xs, ys_on_pol_mv, color="purple")

    plt.plot(xs, ys_off_pol_fv, color="blue")
    plt.plot(xs, ys_off_pol_mv, color="green")
    plt.title("Policy Difference")
    plt.show()
    print(results)
