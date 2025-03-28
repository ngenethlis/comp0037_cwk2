#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math

import matplotlib.pyplot as plt
from analysis_utilities import (get_optimal_policy, matrix_difference_absolute,
                                matrix_if_differ_difference_absolute,
                                policy_to_comparable)
from common.airport_map_drawer import AirportMapDrawer
from common.scenarios import (test_2x2_scenario, test_3x3_scenario,
                              test_two_row_scenario, test_three_row_scenario)
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p1.low_level_actions import LowLevelActionType
from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_policy_drawer import LowLevelPolicyDrawer
from td.sarsa import SARSA

if __name__ == '__main__':
    # airport_map, drawer_height = test_2x2_scenario()
    airport_map, drawer_height = test_three_row_scenario()

    # Show the scenario
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()

    # Create the environment
    env = LowLevelEnvironment(airport_map)

    # Extract the initial policy. This is e-greedy
    pi = env.initial_policy()
    pi.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    pi.set_action(14, 2, LowLevelActionType.MOVE_DOWN)

    ideal_policy_array = policy_to_comparable(pi)

    # Select the controller
    policy_learner = SARSA(env)
    policy_learner.set_initial_policy(pi)

    # These values worked okay for me.
    policy_learner.set_alpha(0.1)
    policy_learner.set_experience_replay_buffer_size(64)
    policy_learner.set_number_of_episodes(32)

    # The drawers for the state value and the policy
    value_function_drawer = ValueFunctionDrawer(policy_learner.value_function(), drawer_height)
    greedy_optimal_policy_drawer = LowLevelPolicyDrawer(policy_learner.policy(), drawer_height)

    error_to_ideal = []

    for i in range(10000):
        print(i)
        policy_learner.find_policy()
        value_function_drawer.update()
        greedy_optimal_policy_drawer.update()
        pi.set_epsilon(1/math.sqrt(1+0.25*i))
        print(f"epsilon={1/math.sqrt(1+i)};alpha={policy_learner.alpha()}")

        policy_array = policy_to_comparable(policy_learner.policy())
        difference_score = matrix_difference_absolute(policy_array, ideal_policy_array)
        error_to_ideal.append(difference_score)

    plt.title('Policy to Error')
    plt.plot(list(range(len(error_to_ideal))), error_to_ideal)
    plt.show()
