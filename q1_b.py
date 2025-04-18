#!/usr/bin/env python3

"""
Created on 7 Mar 2023

@author: steam
"""

import itertools
import multiprocessing as mp
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from analysis_utilities import (get_optimal_policy, matrix_difference_absolute,
                                matrix_if_differ_difference_absolute,
                                policy_to_comparable)
from common.scenarios import test_three_row_scenario
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor
from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from p1.low_level_actions import LowLevelActionType
from p1.low_level_environment import LowLevelEnvironment

FLOAT_EPSILON = 10e-8

class WorkReturnValue(NamedTuple):
    on_value: float
    off_value: float
    on_policy: float
    off_policy: float
    first_visit: bool
    episode_count: int

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

    # Generate ideal policy
    ideal_policy = env.initial_policy() # make a copy
    ideal_policy.set_action(0, 1, LowLevelActionType.MOVE_DOWN_RIGHT)
    ideal_policy.set_action(1, 1, LowLevelActionType.MOVE_DOWN_RIGHT)
    ideal_policy.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    ideal_policy.set_action(14, 2, LowLevelActionType.MOVE_DOWN)
    ideal_policy.set_action(13, 1, LowLevelActionType.MOVE_DOWN_RIGHT)
    ideal_policy.set_action(13, 2, LowLevelActionType.MOVE_DOWN_RIGHT)
    ideal_policy.set_action(12, 2, LowLevelActionType.MOVE_DOWN_RIGHT)

    # Policy evaluation algorithm
    pe = PolicyEvaluator(env)
    pe.set_policy(ideal_policy)
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

    on_policy = policy_to_comparable(airport_map, get_optimal_policy(mcpp.value_function(), env))
    off_policy = policy_to_comparable(airport_map, get_optimal_policy(mcop.value_function(), env))
    ideal_policy = policy_to_comparable(airport_map, ideal_policy)

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
    plt.plot(xs, ys_on_fv, color="red", label='On-Policy First Visit')
    plt.plot(xs, ys_on_mv, color="purple", label='On-Policy Multi Visit')

    plt.plot(xs, ys_off_fv, color="blue", label='Off-Policy First Visit')
    plt.plot(xs, ys_off_mv, color="green", label='Off-Policy Multi Visit')
    plt.legend()
    plt.title("Value Difference")
    plt.savefig('1_b Value Difference.pdf')
    # plt.show()

    # Policy "Difference"
    plt.cla()
    plt.plot(xs, ys_on_pol_fv, color="red", label='On-Policy First Visit')
    plt.plot(xs, ys_on_pol_mv, color="purple", label='On-Policy Multi Visit')

    plt.plot(xs, ys_off_pol_fv, color="blue", label='Off-Policy First-Visit')
    plt.plot(xs, ys_off_pol_mv, color="green", label='Off-Policy Multi-Visit')
    plt.legend()
    plt.title("Policy Difference")
    plt.savefig('1_b Policy Difference.pdf')
    # plt.show()
    # print(results)
