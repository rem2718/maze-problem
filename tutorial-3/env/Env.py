import gymnasium as gym


class Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.state_space = None
        self.action_space = None

    def _get_obs(self):
        pass

    def _get_info(self):
        pass
    
    def _get_reward(self, s):
        pass
    
    def _transition(self, s, a):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, a):
        pass
   
    def close(self):
        pass