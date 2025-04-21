#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

import matplotlib.pyplot as plt
from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer

from td.td_policy_predictor import TDPolicyPredictor
from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

from analysis_utilities import (get_optimal_policy, matrix_difference_absolute,
                                matrix_if_differ_difference_absolute,
                                policy_to_comparable)

class ValueFunctionWrapper:
    def __init__(self, policy_evaluator):
        self._pe = policy_evaluator

    def value(self, x, y):
        return self._pe._v.value(x, y)  # Access internal value function


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
    v_pe = ValueFunctionDrawer(pe.value_function(), drawer_height)
    pe.evaluate()
    v_pe.update()
    # Calling update a second time clears the "just changed" flag
    # which means all the digits will be rendered in black
    v_pe.update()

    # Off policy MC predictors

    #epsilon_b_values = [0.1, 0.2, 0.5, 1.0]

    step = 0.01
    epsilon_b_values = [x*step for x in range(int(1/step))]
    num_values = len(epsilon_b_values)

    mc_predictors = [None] * num_values
    mc_drawers = [None] * num_values

    # Track convergence info
    converged = [False] * num_values
    episodes_to_converge = [None] * num_values

    for i in range(num_values):
        mc_predictors[i] = OffPolicyMCPredictor(env)
        mc_predictors[i].set_use_first_visit(True)
        b = env.initial_policy()
        b.set_epsilon(epsilon_b_values[i])
        mc_predictors[i].set_target_policy(pi)
        mc_predictors[i].set_behaviour_policy(b)
        mc_predictors[i].set_experience_replay_buffer_size(64)
        mc_drawers[i] = ValueFunctionDrawer(mc_predictors[i].value_function(), drawer_height)



    max_episodes = 500


    # Evaluate target policy once using PolicyEvaluator
    pe = PolicyEvaluator(env)
    pe.set_policy(pi)
    pe.evaluate()
    vf_wrapper = ValueFunctionWrapper(pe)

    # Compute the ideal policy just once
    ideal_policy = policy_to_comparable(
        airport_map,
        get_optimal_policy(vf_wrapper, env)
    )

    # Main loop parameters
    max_episodes = 500
    converged = [False] * num_values
    episodes_to_converge = [None] * num_values

    print("=== Start of convergence loop ===")
    for e in range(1, max_episodes + 1):
        print(f"\nEpisode {e}")
        for i in range(num_values):
            if converged[i]:
                continue  # Skip if already converged

            try:
                mc_predictors[i].evaluate()

                # Compare current learned policy to ideal
                learned_policy = policy_to_comparable(
                    airport_map,
                    get_optimal_policy(mc_predictors[i].value_function(), env)
                )

                diff = matrix_difference_absolute(ideal_policy, learned_policy)

                if diff == 0:
                    converged[i] = True
                    episodes_to_converge[i] = e
                    print(f"✅ Converged at epsilon_b = {epsilon_b_values[i]:.2f} (index {i}) in {e} episodes")

            except Exception as ex:
                print(f"❌ Crash at epsilon_b = {epsilon_b_values[i]:.2f} (index {i}) in episode {e}")
                print(f"Exception: {ex}")
                converged[i] = True
                episodes_to_converge[i] = None
         # Plot the result

    print("\n=== Convergence Summary ===")
    for i, eps in enumerate(epsilon_b_values):
        result = (
            f"Converged in {episodes_to_converge[i]} episodes"
            if episodes_to_converge[i] is not None
            else "Crashed or did not converge"
        )

    print(f"epsilon_b = {eps:.2f}: {result}")
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_b_values, episodes_to_converge, marker='o', linestyle='-', color='blue')
    plt.xlabel("Epsilon (behaviour policy)")
    plt.ylabel("Episodes to Converge")
    plt.title("Convergence of Off-Policy MC Predictor vs Epsilon")
    plt.grid(True)
    plt.show()
    plt.savefig('convergence_plot.png')
    v_pe.save_screenshot("q1_c_truth_pe.pdf")
    for i in range(num_values):
        mc_drawers[i].save_screenshot(f"mc-off-{int(epsilon_b_values[i]*10):03}-pe.pdf")
