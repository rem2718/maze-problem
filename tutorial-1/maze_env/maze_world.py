import pygame
import numpy as np
from enum import Enum
import gymnasium as gym
from gymnasium import spaces

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  
        self.window_size = 512
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.agent_image = None
        self.target_image = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0 
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _init_frame(self, pix_square_size):
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        self.agent_image = pygame.transform.scale(pygame.image.load("images/robot.png").convert_alpha(), (pix_square_size/1.1, pix_square_size/1.5))
        self.target_image = pygame.transform.scale(pygame.image.load("images/batteries.png").convert_alpha(), (pix_square_size/1.1, pix_square_size/1.5))
        
    def _render_frame(self):
        pix_square_size =  (self.window_size / self.size)
        
        if self.window is None and self.render_mode == "human":
            self._init_frame(pix_square_size)

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((55, 83 , 104))
        
        canvas.blit(self.target_image, ((self._target_location[0] + 0.05)  * pix_square_size, (self._target_location[1] + 0.15) * pix_square_size))
        canvas.blit(self.agent_image, ((self._agent_location[0] + 0.05)  * pix_square_size, (self._agent_location[1] + 0.15) * pix_square_size))

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (10, 10, 10),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                (10, 10, 10),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
