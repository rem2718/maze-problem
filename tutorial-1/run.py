import gymnasium
import maze_env

from agent.dummy_agent import DummyAgent


env = gymnasium.make("maze_env/Maze-v0", render_mode="human")
agent = DummyAgent(env)

episode_over = False
total_reward = 0

observation, info = env.reset()

while not episode_over:
    action = agent.get_action()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()