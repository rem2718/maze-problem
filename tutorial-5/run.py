import gymnasium

from agent.agent import Agent
import env


env = gymnasium.make("env/Maze-v0", render_mode="human", reg_r=-1)
agent = Agent(env.unwrapped)

total_episodes = 4 

for ep in range(total_episodes):
    state, _ = env.reset(seed=1221)
    episode_over = False
    t = 0  
    values = agent.get_values()
    env.unwrapped.set_values(values)
    while not episode_over:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated
        agent.update(state['agent'], action, reward, next_state['agent'], episode_over)
        state = next_state
        t += 1

    print(f"Episode {ep + 1} finished")

env.close()
