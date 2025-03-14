'''
Created on 8 Mar 2023

@author: steam
'''

import random
import numpy as np

from .td_controller import TDController

# Simplified version of the predictor from S+B

class QLearner(TDController):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDController.__init__(self, environment)

    def initialize(self):
               
        # Set up experience replay buffer
        TDController.initialize(self)
        
        # Change names to change titles on drawn windows
        self._v.set_name("Q-Learning Expected Value Function")
        self._pi.set_name("Q-Learning Greedy Policy")
            
    def _update_action_and_value_functions_from_episode(self, episode):
        
        # This calls a method in the TDController which will update the
        # Q value estimate in the base class and will update
        # the greedy policy and estimated state value function

        # Handle everything up to the last state transition to the terminal state
        s = episode.state(0)
        coords = s.coords()
        reward = episode.reward(0)
        a = episode.action(0)

        for step_count in range(1, episode.number_of_steps()):

            # Q2x: Apply Q-learning to compute / update new_q
            new_q = 0

            # Update the grid
            self._update_q_and_policy(coords, a, new_q)

            # Move to the next step in the episode
            reward = episode.reward(step_count)
            s =  episode.state(step_count)
            coords = s.coords()
            a = episode.action(step_count)

        # Final value
        new_q = reward
        self._update_q_and_policy(coords, a, new_q)
