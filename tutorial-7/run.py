import gymnasium

from agent.agent import Agent
import env


env = gymnasium.make("env/Maze-v0", render_mode="human", reg_r=-1, tar_r=56)
env.reset(seed=1221)
agent = Agent(env.unwrapped)
values = agent.get_values()
env.unwrapped.set_values(values)

state, _ = env.reset(seed=1221)
episode_over = False
t = 0  
while not episode_over:
    action = agent.get_action(state['agent'])
    next_state, reward, terminated, truncated, _ = env.step(action)
    episode_over = terminated or truncated
    agent.update(state['agent'], action, reward, next_state['agent'], episode_over)
    state = next_state
    t += 1

env.close()
