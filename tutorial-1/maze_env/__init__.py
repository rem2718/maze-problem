from gymnasium.envs.registration import register
from maze_env.maze_world import MazeEnv

register(
    id="maze_env/Maze-v0",
    entry_point="maze_env:MazeEnv",
)
