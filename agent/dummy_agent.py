import gymnasium as gym

class DummyAgent:
    def __init__(self, env: gym.Env):
        self.env = env

    def get_action(self) -> int:
        return self.env.action_space.sample()

    def update(self):
        pass
