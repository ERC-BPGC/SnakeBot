from gymnasium.envs.registration import register

register(
    id='SnakeEnv-v0',
    entry_point='my_env.snakebot_env:SnakeEnv',  # Replace 'your_module' with the module name or path
)