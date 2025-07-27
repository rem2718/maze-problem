from gymnasium.envs.registration import register
from env import Env

register(
    id="env/EnvName-v0",
    entry_point="env:EnvName",
)
