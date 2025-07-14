import numpy as np

class MDP():
    def __init__(self, env):
        self.size = env.size
        self.rewards = env.rewards
        self.target = env._target_location
        self.directions = list(env._action_to_direction.values())
        self.values = np.full((self.size, self.size), np.nan)
    
    def _V(self, immediate_r, future_r):
        return future_r + (0.1 * (immediate_r - future_r)) 
   
    def _max_nbr_r(self, s, d, prev_values):
        nbr = np.clip(s + d, 0, self.size - 1)
        prev_val = prev_values[s[0], s[1]]
        new_val = prev_values[nbr[0], nbr[1]]
        return max(new_val, prev_val)
                    
    def _max_r(self, s, prev_values):
        if not np.isnan(prev_values[s[0], s[1]]):
            return None
        
        best_r = -np.inf
        for d in self.directions:
            cur_r = self._max_nbr_r(s, d, prev_values)
            if cur_r > best_r:
                best_r = cur_r

        return best_r if best_r != -np.inf else np.nan
        
    def iterative_values(self):
        eps = 1e-20
        max_iterations = 100
        self.values[self.target[0], self.target[1]] = self.rewards[self.target[0], self.target[1]]
        for _ in range(max_iterations):
            prev_values = np.copy(self.values)
            for s, _ in np.ndenumerate(self.values):
                if np.array_equal(s, self.target):
                    continue
                max_r = self._max_r(s, prev_values)
                if max_r is not None: 
                    self.values[s[0],s[1]] = self._V(self.rewards[s[0], s[1]], max_r)
              
            if np.sum(np.fabs(prev_values - self.values)) <= eps:
                print ('Value-iteration converged at iteration# %d.' %(i+1))
                break
        
        return np.round(self.values, decimals=2)        

