from enum import Enum
import random 

from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pygame

class Actions(Enum):
    up = 0
    down = 1
    left = 2
    right = 3
  
class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode=None, size=8, tar_r=24, reg_r=0, gamma=0.9):
        self.render_mode = render_mode
        self.size = size  
        self.tar_r = tar_r
        self.reg_r = reg_r
        self.gamma = gamma
        self.window_size = 512
        self.rewards = np.full((self.size, self.size), self.reg_r, dtype=float)
        self.values = np.full((self.size, self.size), np.nan, dtype=float)
        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            Actions.up.value: np.array([-1, 0]),
            Actions.down.value: np.array([1, 0]),
            Actions.left.value: np.array([0, -1]),
            Actions.right.value: np.array([0, 1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.agent_image = None
        self.target_image = None

    def _get_obs(self):
        return {"agent": self._agent_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
        
    def _get_reward(self, s):
        x, y  = s
        return self.rewards[x, y]
    
    def _transition(self, s, a): 
        out = False
        dir = self._action_to_direction[a]
        next_s = s + dir
        if next_s[0] < 0 or next_s[0] >= self.size or next_s[1] < 0 or next_s[1] >= self.size:
            out = True
            next_s = np.clip(next_s, 0, self.size - 1)
        return next_s, out


    def _add_obstacles(self):
        for row in range(0, self.size, 2): 
            obstacle_cols = random.sample(range(self.size), k=random.randint(1, self.size - 1))
            self.rewards[row, obstacle_cols] = np.nan

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        self._target_location = np.array([0,0]) 
        self._agent_location = np.array([self.size-1, self.size-1])
        self._add_obstacles()
        self.rewards[self._target_location[0], self._target_location[1]] = self.tar_r 
        self.rewards[self._agent_location[0], self._agent_location[1]] = self.reg_r 
        
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, a):
        loc, out = self._transition(self._agent_location, a) 
        reward = np.nan if out else self._get_reward(loc)
        if not np.isnan(reward):
            self._agent_location = loc

        if self.render_mode == "human":
            self._render_frame()
        
        if self.render_mode == "rgb_array":
            print(self.values)
        
        terminated = np.array_equal(self._agent_location, self._target_location)
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
    def _init_frame(self, pix_square_size):
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", int(pix_square_size // 3))
        self.agent_image = pygame.transform.scale(pygame.image.load("images/robot.png").convert_alpha(), (pix_square_size/1.1, pix_square_size/1.5))
        self.target_image = pygame.transform.scale(pygame.image.load("images/batteries.png").convert_alpha(), (pix_square_size/1.1, pix_square_size/1.5))

    def _render_frame(self):
        block_color = (160, 160, 160)
        grid_color = (55, 83, 104)
        visual_size = self.size + 2 
        pix_square_size = self.window_size / visual_size

        if self.window is None and self.render_mode == "human":
            self._init_frame(pix_square_size)

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(block_color)  

        for row in range(self.size):
            for col in range(self.size):
                if np.isnan(self.rewards[row, col]):
                    continue
                rect = pygame.Rect(
                    (col + 1) * pix_square_size,
                    (row + 1) * pix_square_size,
                    pix_square_size,
                    pix_square_size
                )
                pygame.draw.rect(canvas, grid_color, rect)
                value = self.values[row, col]
                if value == '-' or value is None:
                    continue
                v = value if isinstance(value, str) else str(round(value, 1))
                text = self.font.render(v, True, (255, 255, 255))  
                canvas.blit(text, ((col+1.1) * pix_square_size, (row+1.1) * pix_square_size) )

        canvas.blit(
            self.target_image,
            ((self._target_location[1] + 1 + 0.05) * pix_square_size,
            (self._target_location[0] + 1 + 0.15) * pix_square_size),
        )
        canvas.blit(
            self.agent_image,
            ((self._agent_location[1] + 1.05) * pix_square_size,
            (self._agent_location[0] + 1.15) * pix_square_size),
        )

        for x in range(visual_size + 1):
            pygame.draw.line(
                canvas,
                (10, 10, 10),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                (10, 10, 10),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )
            
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def set_values(self, values):
        self.values = values
