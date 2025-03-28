#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

import matplotlib.pyplot as plt

from analysis_utilities import (matrix_difference_absolute,
                                value_function_to_numpy)
from common.scenarios import test_three_row_scenario
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator
from p1.low_level_actions import LowLevelActionType
from p1.low_level_environment import LowLevelEnvironment
from td.td_policy_predictor import TDPolicyPredictor

if __name__ == '__main__':
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

    width = env.map().width()
    height = env.map().height()
    truth_values = value_function_to_numpy(pe.value_function(), width, height)

    # Range of alpha values
    alpha_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

    num_values = len(alpha_values)

    td_predictors = [None] * num_values
    # td_drawers = [None] * num_values
    td_historical_points = [[] for _ in range(num_values)]

    # TD policy predictor
    for i in range (num_values):
        td_predictors[i] = TDPolicyPredictor(env)
        td_predictors[i].set_experience_replay_buffer_size(64)
        td_predictors[i].set_alpha(alpha_values[i])
        td_predictors[i].set_target_policy(pi)
        # td_drawers[i] = ValueFunctionDrawer(td_predictors[i].value_function(), drawer_height)

    for e in range(400):
        for i in range(num_values):
            td_predictors[i].evaluate()
            values = value_function_to_numpy(td_predictors[i].value_function(), width, height)
            difference = matrix_difference_absolute(values, truth_values)

            td_historical_points[i].append(difference)
            # td_drawers[i].update()

    # v_pe.save_screenshot("truth_pe.pdf")

    # for i in range(num_values):
    #     td_drawers[i].update()
    #     td_drawers[i].save_screenshot(f"td-{int(alpha_values[i]*10):03}-pe.pdf")

    plt.title('Episode Count vs Difference to Truth')
    for i in range(num_values):
        plt.plot(range(len(td_historical_points[i])),
                 td_historical_points[i], label=f'Alpha: {alpha_values[i]}')
    plt.legend(loc='upper right')
    plt.show()
