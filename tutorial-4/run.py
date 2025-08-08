import gymnasium

from agent.agent import Agent
from env import Env


env = gymnasium.make("env/name-v0")
agent = Agent(env)

total_episodes = 100  

for ep in range(total_episodes):
    state, _ = env.reset()
    episode_over = False
    t = 0  

    while not episode_over:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated
        agent.update(state, action, reward, next_state, episode_over)
        state = next_state
        t += 1

    print(f"Episode {ep + 1} finished")

env.close()
