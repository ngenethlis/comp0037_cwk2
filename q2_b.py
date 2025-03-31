#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math

import matplotlib.pyplot as plt
import numpy as np

from analysis_utilities import (get_optimal_policy, matrix_difference_absolute,
                                matrix_if_differ_difference_absolute,
                                policy_to_comparable)
from common.airport_map_drawer import AirportMapDrawer
from common.scenarios import (test_2x2_scenario, test_3x3_scenario,
                              test_three_row_scenario, test_two_row_scenario)
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p1.low_level_actions import LowLevelActionType
from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_policy_drawer import LowLevelPolicyDrawer
from td.q_learner import QLearner

if __name__ == '__main__':
    # airport_map, drawer_height = test_2x2_scenario()
    # airport_map, drawer_height = test_3x3_scenario()
    # airport_map, drawer_height = test_two_row_scenario()
    airport_map, drawer_height = test_three_row_scenario()

    # Show the scenario
    # airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    # airport_map_drawer.update()

    # Create the environment
    env = LowLevelEnvironment(airport_map)

    # Generate ideal policy
    ideal_policy = env.initial_policy() # make a copy
    ideal_policy.set_action(0, 1, LowLevelActionType.MOVE_DOWN_RIGHT)
    ideal_policy.set_action(1, 1, LowLevelActionType.MOVE_DOWN_RIGHT)
    ideal_policy.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    ideal_policy.set_action(14, 2, LowLevelActionType.MOVE_DOWN)
    ideal_policy.set_action(13, 1, LowLevelActionType.MOVE_DOWN_RIGHT)
    ideal_policy.set_action(13, 2, LowLevelActionType.MOVE_DOWN_RIGHT)
    ideal_policy.set_action(12, 2, LowLevelActionType.MOVE_DOWN_RIGHT)

    # Extract the initial policy. This is e-greedy
    pi = env.initial_policy()

    ideal_policy_array = policy_to_comparable(airport_map, ideal_policy)

    # Select the controller
    policy_learner = QLearner(env)
    policy_learner.set_initial_policy(pi)

    # According to Slide 34 of "Temporal Difference Learning", Q-Learning:
    # - (Difficult to get emprical proof of) Convergence is guaranteed if all state and action \
    #   values are visited an infinite number of times
    # - Is Off-Policy => Directly computes the optimal policy.

    # Hence, the best way to validate the approach is to compare the optimal policy
    # with the result we get here.

    # These values worked okay for me.
    policy_learner.set_alpha(0.1)
    policy_learner.set_experience_replay_buffer_size(64)
    policy_learner.set_number_of_episodes(32)

    # The drawers for the state value and the policy
    value_function_drawer = ValueFunctionDrawer(policy_learner.value_function(), drawer_height)
    greedy_optimal_policy_drawer = LowLevelPolicyDrawer(policy_learner.policy(), drawer_height)

    sum_rewards = []
    error_to_ideal = []
    ideal_achieved = 0

    for i in range(10000):
        policy_learner.find_policy()
        value_function_drawer.update()
        greedy_optimal_policy_drawer.update()
        pi.set_epsilon(1/math.sqrt(1+0.25*i))
        print(f"epsilon={1/math.sqrt(1+0.25*i)};alpha={policy_learner.alpha()}")

        policy_array = policy_to_comparable(airport_map, policy_learner.policy())
        sum_rewards.append(np.concatenate(policy_array).sum())
        difference_score = matrix_difference_absolute(policy_array, ideal_policy_array)
        error_to_ideal.append(difference_score)

        if difference_score < 1.0:
            ideal_achieved += 1
            if ideal_achieved == 10:
                print(f'Terminated after {i} iterations')
                break

    value_function_drawer.save_screenshot("2_b Value.pdf")
    greedy_optimal_policy_drawer.save_screenshot("2_b Policy.pdf")

    plt.xlabel('Iterations')
    plt.ylabel('Difference')
    plt.title('Policy Reward vs Ideal Policy Reward')
    plt.plot(list(range(len(error_to_ideal))), error_to_ideal)
    plt.show()

    plt.ylabel('Reward')
    plt.title('Total Reward Curve')
    plt.plot(list(range(len(sum_rewards))), sum_rewards)
    plt.show()
