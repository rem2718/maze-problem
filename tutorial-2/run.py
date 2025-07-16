import gymnasium

from agent.agents import RandomAgent, GreedyAgent, SoftmaxAgent
from maze_env.mdp import MDP
import maze_env


env = gymnasium.make("maze_env/Maze-v0", render_mode="human", size=8, tar_r=20, reg_r=0)
observation, info = env.reset(seed=1221)

mdp = MDP(env.unwrapped)
values = mdp.iterative_values()
env.unwrapped.set_values(values)

agent = RandomAgent(env)
# agent = GreedyAgent(env, values, epsilon=0.1)
# agent = SoftmaxAgent(env, values, temperature=0.1)

episode_over = False

while not episode_over:
    action = agent.get_action(observation["agent"])
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
