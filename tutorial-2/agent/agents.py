import random

import gymnasium as gym
import numpy as np


class RandomAgent:
    def __init__(self, env: gym.Env):
        self.env = env
        
    def get_action(self, _s):
        return self.env.action_space.sample()


class GreedyAgent:
    def __init__(self, env: gym.Env, values, epsilon=0.1):
        self.env = env
        self.values = values
        self.epsilon = epsilon
        self.directions = list(env.unwrapped._action_to_direction.values())
        
    def _get_value(self, s):
        return self.values[s[0], s[1]]
    
    def _best_action(self, s):
        size = self.env.unwrapped.size
        nbrs = [np.clip(s + d, 0, size - 1) for d in self.directions]
        values = [self._get_value(nbr) for nbr in nbrs]
        best_idx = int(np.nanargmax(values))
        return best_idx
    
    def get_action(self, s):
        if random.random() < self.epsilon:
            return self.env.action_space.sample() 
        else:
            return self._best_action(s) 


class SoftmaxAgent:
    def __init__(self, env: gym.Env, values, temperature=0.1):
        self.env = env
        self.values = values
        self.temperature = temperature
        self.directions = list(env.unwrapped._action_to_direction.values())
        
    def _get_value(self, s):
        return self.values[s[0], s[1]]
    
    def _softmax_action(self, s):
        size = self.env.unwrapped.size
        act_num = len(self.directions)
        nbrs = [np.clip(s + d, 0, size - 1) for d in self.directions]
        values = np.array([self._get_value(nbr) for nbr in nbrs])
        values = np.where(np.isnan(values), -np.inf, values)
        
        exp_values = np.exp(values / self.temperature)
        probs = exp_values / np.sum(exp_values)
        action_idx = np.random.choice(act_num, p=probs)
        return action_idx
    
    def get_action(self, s):
        return self._softmax_action(s)