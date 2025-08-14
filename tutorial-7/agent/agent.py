import gymnasium as gym
import numpy as np
        
class Agent:
    def __init__(self, env: gym.Env, theta=1e-10, max_iterations=1000):
        self.env = env
        self.gamma = env.gamma
        self.theta = theta  
        self.max_iterations = max_iterations
        self.rewards = env.unwrapped.rewards
        self.states = [(x, y) for x in range(env.size) for y in range(env.size) if not np.isnan(self._immediate_reward((x, y)))]
        self.actions = list(env.unwrapped._action_to_direction.values())
        self.values = np.full((env.size, env.size), np.nan, dtype=float)
        self.policy = {
            (x, y): [1.0 / env.action_space.n] * env.action_space.n
            for x in range(env.size)
            for y in range(env.size)
        }
        self._iterativeDP()
        
    def _V(self, s):
        v = self.values[s[0], s[1]]
        if np.isnan(v):
            return 0.0
        return v
    
    def _transition(self, s, a):
        return np.clip(s + a, 0, self.env.size - 1)
 
    def _immediate_reward(self, s):
        return self.rewards[s[0], s[1]]
    
    def _cal_optimal_policy(self):
        self.values[self.target[0], self.target[1]] = np.inf
        for s in self.states:
            new_action_probs = np.zeros(self.env.action_space.n)
            if self._immediate_reward(s) == self.env.tar_r: 
                continue 
            best_action = np.nanargmax([
                self.values[s_prime[0], s_prime[1]]
                for s_prime in (self._transition(s, a) for a in self.actions)
            ])
            new_action_probs[best_action] = 1.0
            self.policy[s] = new_action_probs 
    
    def _iterativeDP(self):
        def cal_val(s, a):
            s_prime = self._transition(s, a)
            reward = self._immediate_reward(s_prime)
            if np.isnan(reward):
                return np.nan
            return reward + self.gamma * self._V(s_prime)
        
        for _ in range(self.max_iterations):
            delta = 0
            for s in self.states:
                if self._immediate_reward(s) == self.env.tar_r:
                    self.values[s[0], s[1]] = 0
                    self.target = s
                    continue       
                v = self._V(s)
                self.values[s[0], s[1]] = np.nanmax([cal_val(s, a) for a in self.actions])
                delta = max(delta, abs(v - self.values[s[0], s[1]]))
            if delta < self.theta:
                break
         
        self._cal_optimal_policy()
    
    
    def _policy(self, s):
        return self.policy[s]
    
    def get_action(self, s):
        s = tuple(s) 
        action_probs = self._policy(s)
        return np.random.choice(len(action_probs), p=action_probs)

    def update(self, s, a, r, nxt_s, over):
        pass
    
    def get_values(self):
        return self.values