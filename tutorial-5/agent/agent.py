import gymnasium as gym
import numpy as np
        
class Agent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.i = 0
        self.j = 0
        self.episode = []
        self.returns = np.full((env.size, env.size), 0, dtype=float)
        self.counts = np.full((env.size, env.size), 0, dtype=float)
        self.values = np.full((env.size, env.size), np.nan, dtype=float)
        
    def _V(self, _s=None):
        R = 0
        for s, r in reversed(self.episode):
            x, y = s
            R += r             
            self.returns[x][y] += R
            self.counts[x][y] += 1
            self.values[x][y] = self.returns[x][y] / self.counts[x][y]

        self.episode = []
    
    def _policy(self, _s):
        trajectories = [[2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 0, 0, 0, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 2, 2, 2, 2, 2, 0],
                        [2, 2, 2, 2, 0, 0, 3, 3, 3, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 3, 3, 3, 1, 1, 2, 2, 2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0],
                        [3]
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
        self.episode.append((s, r))
        if over:
            self._V()
    
    def get_values(self):
        return self.values