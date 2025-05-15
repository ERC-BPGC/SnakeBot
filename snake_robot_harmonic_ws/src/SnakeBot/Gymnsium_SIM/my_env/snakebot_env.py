import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import time

class SnakeEnv(MujocoEnv):
    def __init__(self, render_mode='human', 
                 model_path='/home/harikrishnan/ROS_PROJECTS/SnakeBot/snake_robot_harmonic_ws/src/SnakeBot/Gymnsium_SIM/my_env/robot_model/modelV2_with_ground.xml',
                 width=1280, height=720):  # Larger window size
        # Define observation space before initializing the parent class
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32
        )
        
        # Initialize MujocoEnv with model_path, frame_skip, observation_space, render_mode, width, and height
        super().__init__(model_path, frame_skip=1, observation_space=observation_space, 
                        render_mode=render_mode, width=width, height=height)
        
        # Define action space after parent initialization
        self.action_space = gym.spaces.Box(
            low=-5, high=5, shape=(14,), dtype=np.float32
        )

    def _get_obs(self):
        # Extract observation: base state (13) + joint positions (14) + joint velocities (14)
        return np.concatenate([
            self.data.qpos[:7],   # Base position (3) and orientation quaternion (4)
            self.data.qvel[:6],   # Base linear velocity (3) and angular velocity (3)
            self.data.qpos[7:21], # Positions of the 14 actuated joints
            self.data.qvel[6:20]  # Velocities of the 14 actuated joints
        ])

    def step(self, action):
        # Apply actions to actuators and simulate one step
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self.data.qvel[0]  # Reward: x-velocity of the base
        done = False  # No early termination; use fixed episode length
        return obs, reward, done, {}

    def reset_model(self):
        # Reset to initial state defined in XML
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

# Test the environment with continuous loop and larger window
def main():
    try:
        # Create environment with human rendering and larger window
        env = SnakeEnv(render_mode='human', width=1280, height=720)
        obs = env.reset()
        
        print("Simulation running. Press 'Q' to exit or close the window.")

        while True:
            

            # Step the environment with a random action
            action = env.action_space.sample()  # Random actions
            obs, reward, done, _ = env.step(action)
            
            # Render the simulation
            env.render()

            # Small delay to control simulation speed and reduce CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Exiting: Keyboard interrupt (Ctrl+C).")
    except Exception as e:
        print(f"Exiting: An error occurred: {e}")
    finally:
        # Clean up and close the environment
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()