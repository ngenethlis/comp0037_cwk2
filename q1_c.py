#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from analysis_utilities import policy_to_comparable, get_optimal_policy, matrix_difference_absolute

from common.scenarios import test_three_row_scenario
from common.airport_map import AirportMap
from common.airport_map_drawer import AirportMapDrawer

from td.td_policy_predictor import TDPolicyPredictor
from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

# if the policy differs from the ideal by this much, we accept anyway
POLICY_THRESHOLD = 5
MAXIMUM_EPISODE_COUNT = 50_000
# MAXIMUM_EPISODE_COUNT = 100

def work(input_tuple: tuple[OffPolicyMCPredictor, AirportMap, LowLevelEnvironment, np.ndarray[np.ndarray[int]]]) -> tuple[OffPolicyMCPredictor, int]:
    predictor, airport_map, env, comparable_ideal = input_tuple

    for ep in range(MAXIMUM_EPISODE_COUNT):
        predictor.evaluate()
        current_policy = policy_to_comparable(
            airport_map, get_optimal_policy(predictor.value_function(), env))
        difference = matrix_difference_absolute(comparable_ideal, current_policy)
        if -POLICY_THRESHOLD <= difference <= POLICY_THRESHOLD:
            return predictor, ep
    return predictor, MAXIMUM_EPISODE_COUNT

if __name__ == '__main__':
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
    comparable_ideal_policy = policy_to_comparable(airport_map, ideal_policy)

    # Policy evaluation algorithm
    pe = PolicyEvaluator(env)
    pe.set_policy(pi)
    # v_pe = ValueFunctionDrawer(pe.value_function(), drawer_height)
    pe.evaluate()
    # v_pe.update()
    # Calling update a second time clears the "just changed" flag
    # which means all the digits will be rendered in black
    # v_pe.update()

    # Off policy MC predictors

    epsilon_b_values = [x*0.1 for x in range(11)]

    num_values = len(epsilon_b_values)

    mc_predictors = [None] * num_values
    mc_drawers = [None] * num_values

    for i in range(num_values):
        mc_predictors[i] = OffPolicyMCPredictor(env)
        mc_predictors[i].set_use_first_visit(True)
        b = env.initial_policy()
        b.set_epsilon(epsilon_b_values[i])
        mc_predictors[i].set_target_policy(pi)
        mc_predictors[i].set_behaviour_policy(b)
        mc_predictors[i].set_experience_replay_buffer_size(64)
        # mc_drawers[i] = ValueFunctionDrawer(mc_predictors[i].value_function(), drawer_height)

    with multiprocessing.Pool() as p:
        results = p.map(
            work,
            [(mc_predictors[i],
              airport_map, env,
              comparable_ideal_policy)
             for i in range(num_values)])

    converged_episodes = [t[1] for t in results]
    predictors = [t[0] for t in results]

    plt.plot(epsilon_b_values, converged_episodes)
    plt.title('Epsilon value to converged episode count')
    plt.savefig('1_c Epsilon Value Converged Episode Count.pdf')

    for idx, predictor in enumerate(predictors):
        policy = get_optimal_policy(predictor.value_function(), env)

        drawer = LowLevelPolicyDrawer(policy, drawer_height)
        drawer.update()
        drawer.save_screenshot(f'mc-off-{idx}-policy.pdf')

    # v_pe.save_screenshot("q1_c_truth_pe.pdf")
    # for i in range(num_values):
    #     mc_drawers[i].save_screenshot(f"mc-off-{int(epsilon_b_values[i]*10):03}-pe.pdf")
