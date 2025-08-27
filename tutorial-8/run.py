import gymnasium
import time

from agent.agent import Agent
import env


env = gymnasium.make("env/Maze-v0", render_mode=None, reg_r=-1, tar_r=56)
env.reset(seed=1221)
agent = Agent(env.unwrapped, first_visit=False, ars=True)
values = agent.get_values()
env.unwrapped.set_values(values)

state, _ = env.reset(seed=1221)
episode_over = False
max_iterations = 1000
t = 0

for i in range(max_iterations):
    env.reset(seed=1221)
    while not episode_over:
        action = agent.get_action(state['agent'])
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated
        policy_stable = agent.update(state['agent'], action, reward, next_state['agent'], episode_over)
        state = next_state
        t += 1
    
    print(f"Episode {i+1} finished after {t+1} timesteps")
    episode_over = False
    t = 0
    if policy_stable:
        print(f"Policy converged after {i+1} episodes")
        break

env = gymnasium.make("env/Maze-v0", render_mode="human", reg_r=-1, tar_r=56)
env.reset(seed=1221)
env.unwrapped.set_values(values)
while not episode_over:
    action = agent.get_action(state['agent'])
    next_state, reward, terminated, truncated, _ = env.step(action)
    episode_over = terminated or truncated
    agent.update(state['agent'], action, reward, next_state['agent'], episode_over)
    state = next_state

env.close()
