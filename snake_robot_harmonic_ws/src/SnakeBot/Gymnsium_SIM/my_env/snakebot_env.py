import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import time
import matplotlib.pyplot as plt

class SnakeEnv(MujocoEnv):
    def __init__(self, render_mode='human', 
                 model_path='/home/harikrishnan/ROS_PROJECTS/SnakeBot/snake_robot_harmonic_ws/src/SnakeBot/Gymnsium_SIM/my_env/robot_model/modelV2_with_ground.xml',
                 width=1280, height=720):
        # Define observation space
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32
        )
        
        # Initialize MujocoEnv
        super().__init__(model_path, frame_skip=1, observation_space=observation_space, 
                        render_mode=render_mode, width=width, height=height)
        
        # Snake properties
        self.num_joints = 14  # 7 vertical + 7 horizontal
        self.num_modules = 7  # Each module has 1 vertical + 1 horizontal joint
        
        # Oscillator parameters (will be set on reset)
        self.parameters = None
    
    def _get_obs(self):
        # Observation: base state (13) + joint positions (14) + joint velocities (14)
        return np.concatenate([
            self.data.qpos[:7],   # Base position (3) + quaternion (4)
            self.data.qvel[:6],   # Base linear (3) + angular (3) velocities
            self.data.qpos[7:21], # 14 joint positions
            self.data.qvel[6:20]  # 14 joint velocities
        ])
    
    def step(self):
        # Get oscillator parameters
        if self.parameters is None:
            self._set_random_parameters()
        amp_v, freq_v, offset_v, phase_shift_v = self.parameters[0:4]
        amp_h, freq_h, offset_h, phase_shift_h = self.parameters[4:8]
        
        # Current simulation time
        t = self.data.time
        
        # Compute sinusoidal control values
        control_values = np.zeros(self.num_joints)
        for i in range(self.num_modules):
            # Vertical joint (0, 2, 4, ..., 12)
            phase_v = i * phase_shift_v
            control_values[2*i] = amp_v * np.sin(2 * np.pi * freq_v * t + phase_v) + offset_v
            # Horizontal joint (1, 3, 5, ..., 13)
            phase_h = i * phase_shift_h
            control_values[2*i + 1] = amp_h * np.sin(2 * np.pi * freq_h * t + phase_h) + offset_h
        
        # Apply control values (clamped to [-5, 5] per model’s ctrlrange)
        control_values = np.clip(control_values, -5, 5)
        self.do_simulation(control_values, self.frame_skip)
        
        obs = self._get_obs()
        reward = self.data.qvel[0]  # Base x-velocity
        done = self.data.time >= 10.0  # 10-second episodes
        return obs, reward, done, {}
    
    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        self._set_random_parameters()  # New parameters on reset
        return self._get_obs()
    
    def _set_random_parameters(self):
        # Random oscillator parameters: [amp_v, freq_v, offset_v, phase_shift_v, amp_h, freq_h, offset_h, phase_shift_h]
        self.parameters = np.array([
            np.random.uniform(0, 1),      # amp_v
            np.random.uniform(0.1, 1),    # freq_v
            np.random.uniform(-1, 1),     # offset_v
            np.random.uniform(-np.pi, np.pi),  # phase_shift_v
            np.random.uniform(0, 1),      # amp_h
            np.random.uniform(0.1, 1),    # freq_h
            np.random.uniform(-1, 1),     # offset_h
            np.random.uniform(-np.pi, np.pi)   # phase_shift_h
        ])

def plot_rewards(total_rewards, episode_rewards_list=None):
    # Plot total rewards per episode
    plt.figure(figsize=(10, 5))
    episodes = range(1, len(total_rewards) + 1)
    plt.plot(episodes, total_rewards, marker='o', linestyle='-', color='b')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.grid(True)
    plt.savefig('total_rewards_plot.png')
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
    
    # Optionally plot per-step rewards for the last episode
    if episode_rewards_list and episode_rewards_list[-1]:
        plt.figure(figsize=(10, 5))
        steps = range(1, len(episode_rewards_list[-1]) + 1)
        plt.plot(steps, episode_rewards_list[-1], linestyle='-', color='r')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Per-Step Reward in Last Episode')
        plt.grid(True)
        plt.savefig('last_episode_rewards_plot.png')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

def main():
    try:
        env = SnakeEnv(render_mode='human', width=1280, height=720)
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        episode_rewards = []
        total_rewards = []
        episode_rewards_list = []
        episode_count = 0
        
        # Parameter names
        param_names = ['amp_v', 'freq_v', 'offset_v', 'phase_shift_v', 
                      'amp_h', 'freq_h', 'offset_h', 'phase_shift_h']
        print("Simulation running. Close the window or press Ctrl+C to exit.")
        print("New episode started with parameters:")
        for name, value in zip(param_names, env.parameters):
            print(f"  {name}: {value:.4f}")
        
        while True:
            if done:
                # Store episode data
                avg_reward = total_reward / step_count if step_count > 0 else 0
                total_rewards.append(total_reward)
                episode_rewards_list.append(episode_rewards)
                episode_count += 1
                
                # Print episode summary
                print(f"\nEpisode {episode_count} finished. Total Reward: {total_reward:.4f}, Average Reward: {avg_reward:.4f}")
                
                obs = env.reset()
                done = False
                total_reward = 0.0
                step_count = 0
                episode_rewards = []
                
                # Print new parameters
                print(f"\nNew episode {episode_count + 1} started with parameters:")
                for name, value in zip(param_names, env.parameters):
                    print(f"  {name}: {value:.4f}")
            
            obs, reward, done, _ = env.step()
            env.render()
            
            # Accumulate rewards
            total_reward += reward
            step_count += 1
            episode_rewards.append(reward)
            
            # Print current reward
            print(f"Episode {episode_count + 1}, Step {step_count}: Reward: {reward:.4f}", end='\r')
            
            time.sleep(0.01)  # ~100 FPS
        
    except KeyboardInterrupt:
        print("\nExiting: Keyboard interrupt (Ctrl+C).")
    except Exception as e:
        print(f"\nExiting: An error occurred: {e}")
    finally:
        # Store final episode data if any
        if step_count > 0:
            total_rewards.append(total_reward)
            episode_rewards_list.append(episode_rewards)
            avg_reward = total_reward / step_count if step_count > 0 else 0
            print(f"\nFinal Episode {episode_count + 1} finished. Total Reward: {total_reward:.4f}, Average Reward: {avg_reward:.4f}")
        
        env.close()
        print("Environment closed.")
        
        # Plot rewards
        if total_rewards:
            plot_rewards(total_rewards, episode_rewards_list)
        else:
            print("No episodes completed, skipping plotting.")

if __name__ == "__main__":
    main()