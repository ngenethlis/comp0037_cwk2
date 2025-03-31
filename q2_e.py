#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math

import matplotlib.pyplot as plt

from common.scenarios import corridor_scenario

from common.airport_map_drawer import AirportMapDrawer

from numpy import cumsum


from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':
    airport_map, drawer_height = corridor_scenario()

    # Show the scenario        
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()
    
    # Create the environment
    env = LowLevelEnvironment(airport_map)

    # Extract the initial policy. This is e-greedy
    pi = env.initial_policy()
    
    # Select the controller
    policy_learner = QLearner(env)   
    policy_learner.set_initial_policy(pi)

    # These values worked okay for me.
    policy_learner.set_alpha(0.1)
    policy_learner.set_experience_replay_buffer_size(64)
    policy_learner.set_number_of_episodes(32)
    
    # The drawers for the state value and the policy
    value_function_drawer = ValueFunctionDrawer(policy_learner.value_function(), drawer_height)    
    greedy_optimal_policy_drawer = LowLevelPolicyDrawer(policy_learner.policy(), drawer_height)

    timings = []
    timings2 = []

    for i in range(100):
        print(i)
        lt, lt2 = policy_learner.find_policy()
        timings += lt
        timings2 += lt2

        value_function_drawer.update()
        greedy_optimal_policy_drawer.update()
        pi.set_epsilon(1/math.sqrt(1+0.25*i))

        print(f"epsilon={1/math.sqrt(1+i)};alpha={policy_learner.alpha()}")

    timings = cumsum(timings)

    plt.plot(range(len(timings)), timings)
    plt.title("Cumulative Time Taken Per Episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Cumulative Time Taken (ns)")
    plt.show()

    timings2 = sorted(timings2, key=lambda x: x[1])
    y = [a[0] for a in timings2]
    x = [a[1] for a in timings2]
    plt.plot(x, y)
    plt.title("Time Taken Against Number of Steps")
    plt.xlabel("Number of Steps")
    plt.ylabel("Time Taken (ns)")
    plt.show()

