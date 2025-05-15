import gymnasium as gym
import my_env  # this triggers the registration

env = gym.make("SnakeBot-v0", render_mode="human")
obs, _ = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
