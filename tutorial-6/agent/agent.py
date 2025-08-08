import gymnasium as gym
import numpy as np
        
class Agent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.gamma = self.env.gamma
        self.i = 0
        self.j = 0
        self.values = np.full((env.size, env.size), np.nan, dtype=float)
        
    def _V(self, s):
        v = self.values[s[0], s[1]]
        if np.isnan(v):
            return 0.0
        return v
     
    
    def _policy(self, _s):
        trajectories = [[2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 0, 0, 0, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 2, 2, 2, 2, 2, 0],
                        [2, 2, 2, 2, 0, 0, 3, 3, 3, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 3, 3, 3, 1, 1, 2, 2, 2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0],
                        [3, 3]
                        ]
        
        action = trajectories[self.j][self.i]
        if self.i < len(trajectories[self.j]) - 1:
            self.i +=1
        else:
            self.i = 0
            self.j += 1
        return action
    
    def get_action(self, s):
        return self._policy(s)

    def update(self, s, a, r, nxt_s, over):
        print(r)
        self.values[s[0], s[1]] = r + self.gamma * self._V(nxt_s)
    
    def get_values(self):
        return self.values