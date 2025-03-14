#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

import multiprocessing as mp
import itertools
import numpy as np
import matplotlib.pyplot as plt

from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer

from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

def matrix_difference_absolute(matrix_1: np.ndarray, matrix_2: np.ndarray) -> float:
    return np.linalg.norm(np.nan_to_num(matrix_1) - np.nan_to_num(matrix_2))

def work(args: tuple[bool, int]) -> tuple[float, float, bool, int]:
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

    mcpp_values = np.array([[mcpp.value_function().value(x, y) for x in range(env.map().width())] for y in range(env.map().height())])
    mcop_values = np.array([[mcop.value_function().value(x, y) for x in range(env.map().width())] for y in range(env.map().height())])
    pe_values = np.array([[pe.value_function().value(x, y) for x in range(env.map().width())] for y in range(env.map().height())])
    return (matrix_difference_absolute(mcpp_values, pe_values),
            matrix_difference_absolute(mcop_values, pe_values),
            first_visit,
            episode_count)

if __name__ == '__main__':
    MAX_EPISODE_COUNT = 100
    with mp.Pool() as pool:
        results = pool.map(work, itertools.product([True, False], list(range(0, MAX_EPISODE_COUNT + 1))))


    ys_on_fv = [r[0] for r in sorted([(result[1], result[3]) for result in results if result[2]], key=lambda t: t[1])]
    ys_off_fv = [r[0] for r in sorted([(result[0], result[3]) for result in results if result[2]], key=lambda t: t[1])]
    ys_on_mv = [r[0] for r in sorted([(result[1], result[3]) for result in results if not result[2]], key=lambda t: t[1])]
    ys_off_mv = [r[0] for r in sorted([(result[0], result[3]) for result in results if not result[2]], key=lambda t: t[1])]

    xs = list(range(0, MAX_EPISODE_COUNT + 1))
    plt.plot(xs, ys_on_fv, color='red')
    plt.plot(xs, ys_on_mv, color='purple')

    plt.plot(xs, ys_off_fv, color='blue')
    plt.plot(xs, ys_off_mv, color='green')
    plt.show()
    print(results)
