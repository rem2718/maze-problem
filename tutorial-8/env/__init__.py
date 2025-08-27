from gymnasium.envs.registration import register
from env.Env import MazeEnv

register(
    id="env/Maze-v0",
    entry_point="env.Env:MazeEnv",
)
