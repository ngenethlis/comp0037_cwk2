'''
Created on 8 Mar 2023

@author: ucacsjj
'''

import numpy as np

from .td_controller import TDController

# Simplified version of the predictor from S+B

class SARSA(TDController):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDController.__init__(self, environment)

    def initialize(self):
        
        TDController.initialize(self)
        
        self._v.set_name("SARSA Expected Value Function")
        self._pi.set_name("SARSA Greedy Policy")
                    
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
            # Q2x: Apply SARSA to compute / update new_q
            x, y = coords
            A = a
            self._environment.reset(s)
            S_prime, R, _, _, _ = self._environment.step(A)

            x_prime, y_prime = S_prime.coords()
            A_prime = self._pi.action(x, y)

            new_q = self._Q[x, y, A] + self.alpha() * (R + self.gamma() * self._Q[x_prime, y_prime, A_prime] - self._Q[x, y, A])
           
            # Update the grid
            self._update_q_and_policy(coords, A, new_q)

            # Move to the next step in the episode
            reward = R
            s = S_prime
            coords = s.coords()
            a = A_prime

            if S_prime.is_terminal():
                break
