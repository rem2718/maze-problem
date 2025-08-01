from gymnasium.envs.registration import register
from env import Env

register(
    id="maze_env/Maze-v0",
    entry_point="maze_env:MazeEnv",
)
