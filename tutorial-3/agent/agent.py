import gymnasium as gym


class Agent:
    def __init__(self, env: gym.Env):
        self.env = env
        
    def _Q(self, s, a):
        pass
    
    #or
    def _V(self, s):
        pass
    
    def _policy(self, s):
        pass
    
    def get_action(self, s):
        pass

    def update(self, s, a, r, nxt_s, over):
        pass
