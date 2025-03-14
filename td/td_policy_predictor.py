'''
Created on 8 Mar 2023

@author: ucacsjj
'''

from monte_carlo.episode_sampler import EpisodeSampler

from .td_algorithm_base import TDAlgorithmBase

class TDPolicyPredictor(TDAlgorithmBase):

    def __init__(self, environment):
        
        TDAlgorithmBase.__init__(self, environment)
        
        self._minibatch_buffer= [None]
                
    def set_target_policy(self, policy):        
        self._pi = policy        
        self.initialize()
        self._v.set_name("TDPolicyPredictor")
        
    def evaluate(self):
        
        episode_sampler = EpisodeSampler(self._environment)
        
        for episode in range(self._number_of_episodes):

            # Choose the start for the episode            
            start_x, start_a  = self._select_episode_start()
            self._environment.reset(start_x) 
            
            # Now sample it
            new_episode = episode_sampler.sample_episode(self._pi, start_x, start_a)

            # If we didn't terminate, skip this episode
            if new_episode.terminated_successfully() is False:
                continue
            
            # Update with the current episode
            self._update_value_function_from_episode(new_episode)
            
            # Pick several randomly from the experience replay buffer and update with those as well
            for _ in range(min(self._replays_per_update, self._stored_experiences)):
                episode = self._draw_random_episode_from_experience_replay_buffer()
                self._update_value_function_from_episode(episode)
                
            self._add_episode_to_experience_replay_buffer(new_episode)
            
    def _update_value_function_from_episode(self, episode):

        # Q1e:
        # Complete implementation of this method
        # Each time you update the state value function, you will need to make a
        # call of the form:
        #
        # self._v.set_value(x_cell_coord, y_cell_coord, new_v)

        self._environment.reset(episode.state(0))

        for i in range(episode.number_of_steps()):
            if episode.state(i).is_terminal():
                break

            x, y = episode.state(i).coords()

            A = self._pi.action(x, y)
            self._environment.reset(episode.state(i))
            S_prime, R, _, _, _ = self._environment.step(A)
            sx, sy = S_prime.coords()
            self._v.set_value(x, y, self._v.value(x, y) + self.alpha() * (R + self.gamma() * self._v.value(sx, sy) - self._v.value(x, y)))

