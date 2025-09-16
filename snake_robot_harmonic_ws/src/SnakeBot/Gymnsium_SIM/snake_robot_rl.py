# Snake Robot Reinforcement Learning Implementation
# Comprehensive RL setup for snake robot locomotion in MuJoCo with Gymnasium

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco_viewer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import xml.etree.ElementTree as ET


class SnakeRobotEnv(gym.Env):
    """
    Custom Gymnasium environment for snake robot locomotion control.
    Supports forward, backward, left, and right movement commands.
    """
    
    def __init__(self, model_path="/home/harikrishnan/ROS_PROJECTS/snake_robot_harmonic_ws/snake_robot_harmonic_ws/src/SnakeBot/Gymnsium_SIM/my_env/robot_model/modelV2_with_ground.xml", render_mode=None, max_episode_steps=1000):
        super().__init__()
        
        # Load MuJoCo model
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Rendering setup
        self.render_mode = render_mode
        self.viewer = None
        
        # Episode management
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Get model information
        self.n_joints = self.model.nq - 7  # Exclude base position and orientation
        self.n_actuators = self.model.nu
        
        # Action space: continuous control for joint actuators
        self.action_space = spaces.Box(
        low=np.array([0, 0.1, 0, -1, 0, 0.1, 0, -1]),  # sensor lower bounds for each param
        high=np.array([1, 2, np.pi, 1, 1, 2, np.pi, 1]),  # upper bounds
        dtype=np.float32
    )
        # Observation space: joint positions, velocities, body position, orientation
        obs_dim = self.n_joints * 2 + 6 + 3  # joints pos/vel + base pose + target direction
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Movement targets
        self.movement_commands = {
            'forward': np.array([1.0, 0.0]),
            'backward': np.array([-1.0, 0.0]),
            'left': np.array([0.0, 1.0]),
            'right': np.array([0.0, -1.0])
        }
        
        # Current target direction
        self.target_direction = np.array([1.0, 0.0])  # Default: forward
        
        # Initialize tracking variables
        self.initial_position = None
        self.previous_position = None
        self.previous_time = 0
        self.cpg_hold_steps = 100  # or 5–20, depending on simulation frequency
        self.cpg_counter = 0
        self.current_cpg_action = np.zeros(self.action_space.shape)

        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cpg_counter = 0
        self.current_cpg_action = np.zeros(self.action_space.shape)
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Add small random perturbations to initial joint positions
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize joint positions with small random values
        joint_range = 0.1  # radians
        for i in range(7, self.model.nq):  # Skip base position/orientation
            self.data.qpos[i] = np.random.uniform(-joint_range, joint_range)
        
        # Random target direction for varied training
        if options and 'target_direction' in options:
            direction_key = options['target_direction']
            self.target_direction = self.movement_commands[direction_key]
        else:
            # Randomly choose direction for training diversity
            direction_keys = list(self.movement_commands.keys())
            chosen_direction = np.random.choice(direction_keys)
            self.target_direction = self.movement_commands[chosen_direction]
        
        # Forward step to initialize physics
        mujoco.mj_forward(self.model, self.data)
        
        # Store initial state
        self.initial_position = self.data.qpos[:3].copy()
        self.previous_position = self.initial_position.copy()
        self.previous_time = self.data.time
        self.current_step = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = {'target_direction': self.target_direction}
        
        return observation, info
    
    def step(self, action):
        # Apply action to actuators with scaling
         # Only update CPG parameters every cpg_hold_steps
        if self.cpg_counter % self.cpg_hold_steps == 0:
            self.current_cpg_action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self.cpg_counter += 1
        cpg_params = self.current_cpg_action
        amplitude1, frequency1, phase1, offset1, amplitude2, frequency2, phase2, offset2 = cpg_params
        Debug=False    

        # Debug print every step (or every N steps if too spammy)
        if self.current_step % 50 == 0 and Debug:   # change 50 → 1 if you want every step
            print(f"[DEBUG] Step {self.current_step} | "
                f"A1={amplitude1:.3f}, F1={frequency1:.3f}, P1={phase1:.3f}, O1={offset1:.3f} | "
                f"A2={amplitude2:.3f}, F2={frequency2:.3f}, P2={phase2:.3f}, O2={offset2:.3f}")
    # Use CPG parameters to compute oscillatory joint torques internally
    # For example:
    # - Generate phase for each joint based on CPG oscillator equations and action parameters
    # - Compute sine wave outputs for each joint with amplitude, frequency, and phase offsets
    # - Assign these as torques to self.data.ctrl[:]
    
    # Example simplified:
        t = self.data.time
        for j in range(self.n_actuators):
            wave_type = j % 2  # horizontal / vertical mapping (or use your own logic)
            amplitude = cpg_params[0] if wave_type == 0 else cpg_params[4]
            frequency = cpg_params[1] if wave_type == 0 else cpg_params[5]
            phase_shift = cpg_params[2] if wave_type == 0 else cpg_params[6]
            offset = cpg_params[3] if wave_type == 0 else cpg_params[7]

            phase = (t * frequency * 2 * np.pi) + phase_shift
            torque = amplitude * np.sin(phase) + np.sin(offset*t)
            self.data.ctrl[j] = torque
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        # Update previous position for next step
        self.previous_position = self.data.qpos[:3].copy()
        self.previous_time = self.data.time
        
        info = {
            'position': self.data.qpos[:3].copy(),
            'velocity': self._get_velocity(),
            'target_direction': self.target_direction
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation state"""
        # Joint positions (excluding base position/orientation)
        joint_pos = self.data.qpos[7:].copy()
        
        # Joint velocities
        joint_vel = self.data.qvel[6:].copy()  # Skip base linear/angular velocity
        
        # Base position and orientation
        base_pos = self.data.qpos[:3].copy()
        base_quat = self.data.qpos[3:7].copy()
        
        # Convert quaternion to euler angles for orientation representation
        base_euler = self._quat_to_euler(base_quat)
        
        # Target direction (normalized)
        target_dir = self.target_direction.copy()
        target_dir_3d = np.array([target_dir[0], target_dir[1], 0.0])
        
        # Combine all observations
        obs = np.concatenate([
            joint_pos,
            joint_vel, 
            base_pos,
            base_euler,
            target_dir_3d
        ])
        
        return obs.astype(np.float32)
    
    def _compute_reward(self):
        """Compute reward based on movement in target direction"""
        # Current position
        current_pos = self.data.qpos[:3]
        
        # Calculate displacement since last step
        displacement = current_pos[:2] - self.previous_position[:2]
        threshold = 0.005  # 0.5 cm in meters

           
        dt = self.data.time - self.previous_time
        
        if dt > 0:
            velocity_2d =self.data.subtree_linvel[1][:2]
            if 1:
                print(velocity_2d)
        else:
            velocity_2d = np.zeros(2)
        
        # Reward for moving in target direction
        direction_reward = np.dot(velocity_2d, self.target_direction) * 70.0
        
        # Penalty for excessive joint movements (encourage smooth motion)
        joint_vel = self.data.qvel[6:]  # Skip base velocities
        smoothness_penalty = -0.01 * np.sum(np.square(joint_vel))
     
        
        # Stability reward (penalize excessive roll/pitch)
        base_orientation = self._quat_to_euler(self.data.qpos[3:7])
        roll, pitch = base_orientation[0], base_orientation[1]
        stability_penalty = -0.5 * (roll**2 + pitch**2)
        
        # Height penalty (encourage staying on ground)
        
        
        # Combine rewards
        total_reward = (direction_reward)+stability_penalty
        
        return total_reward
    
    def _is_terminated(self):
        """Check if episode should be terminated"""
        # Check if robot has fallen over (excessive roll/pitch)
        base_orientation = self._quat_to_euler(self.data.qpos[3:7])
        roll, pitch = abs(base_orientation[0]), abs(base_orientation[1])
        
        if roll > np.pi/3 or pitch > np.pi/3:  # 60 degrees
            return True
        
        # Check if robot is too high (lost contact with ground)
        height = self.data.qpos[2]
        if height > 0.5:  # Adjust based on your robot size
            return True
            
        return False
    
    def _get_velocity(self):
        """Get current velocity of the robot base"""
        return self.data.qvel[:3].copy()
    
    def _quat_to_euler(self, quat):
        """Convert quaternion to euler angles (roll, pitch, yaw)"""
        # Assuming quaternion format: [w, x, y, z]
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.render()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Deep Q-Network (DQN) for discrete actions
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 256, 128]):
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


# Actor-Critic Networks for PPO
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 256]):
        super(Actor, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Mean and log_std for continuous actions
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
    def forward(self, state):
        x = self.network(state)
        mean = torch.tanh(self.mean_layer(x))  # Bounded to [-1, 1]
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        std = torch.exp(log_std)
        
        return mean, std


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dims=[512, 256]):
        super(Critic, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state)


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Memory
        self.memory = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, std = self.actor(state)
            dist = torch.distributions.Normal(mean, std)
            z = dist.rsample()
            squashed = torch.tanh(z)
            action_low = torch.tensor([0, 0.1, 0, -1, 0, 0.1, 0, -1], device=self.device)
            action_high = torch.tensor([1, 2, np.pi, 1, 1, 2, np.pi, 1], device=self.device)
            action = action_low + (squashed + 1.0) * 0.5 * (action_high - action_low)
            log_prob = dist.log_prob(z) - torch.log(1 - squashed.pow(2) + 1e-6)

            action_log_prob =          log_prob.sum(dim=-1)

            
        return action.cpu().numpy().flatten(), action_log_prob.cpu().numpy()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))
    
    def update(self):
        if len(self.memory) == 0:
            return
       
# Convert list of numpy arrays to numpy arrays first, then to tensor
        states = torch.FloatTensor(np.array([t[0] for t in self.memory])).to(self.device)
        actions = torch.FloatTensor(np.array([t[1] for t in self.memory])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in self.memory])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in self.memory])).to(self.device)
        dones = torch.BoolTensor(np.array([t[4] for t in self.memory])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array([t[5] for t in self.memory])).to(self.device)
        # Calculate returns and advantages
        returns = []
        advantages = []
        
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            
            returns_tensor = torch.zeros_like(rewards)
            advantages_tensor = torch.zeros_like(rewards)
            
            # Calculate returns (discounted future rewards)
            running_return = 0
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    running_return = 0
                running_return = rewards[t] + self.gamma * running_return
                returns_tensor[t] = running_return
            
            # Calculate advantages using GAE
            running_advantage = 0
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    running_advantage = 0
                    delta = rewards[t] - values[t]
                else:
                    delta = rewards[t] + self.gamma * next_values[t] - values[t]
                
                running_advantage = delta + self.gamma * 0.95 * running_advantage  # GAE lambda = 0.95
                advantages_tensor[t] = running_advantage
            
            # Normalize advantages
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Actor update
            mean, std = self.actor(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Critic update
            value_pred = self.critic(states).squeeze()
            critic_loss = F.mse_loss(value_pred, returns_tensor)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # Clear memory
        self.memory = []
    
    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


def train_snake_robot(model_path="modelV2_with_ground.xml", 
                     num_episodes=5000, 
                     save_interval=100,
                     render_training=False):
    """
    Main training function for the snake robot RL agent
    """
    
    # Create environment
    env = SnakeRobotEnv(model_path=model_path, 
                       render_mode="human" if render_training else None)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent
    agent = PPOAgent(state_dim, action_dim)
    action_low  = np.array([0, 0.1, 0, -1, 0, 0.1, 0, -1])
    action_high = np.array([1, 2, np.pi, 1, 1, 2, np.pi, 1])
    fixed_cpg_params = np.array([0.6872, 0.3809, 0.6147, 0.4442,
                                0.6888, 0.2121, 1.2208, -0.5433])
   
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    
    # Create saves directory
    os.makedirs("snake_robot_models", exist_ok=True)
    
    print("Starting training...")
    
    for episode in range(num_episodes):
        # Randomly select target direction for diverse training
        directions = ['forward', 'backward', 'left', 'right']
        target_direction ='forward'#TODO

        
        state, info = env.reset(options={'target_direction': target_direction})
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action
            if episode_length == 0:
                action = fixed_cpg_params.copy()
                log_prob = np.zeros(1)  # Dummy log_prob
            else:
                action, log_prob = agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done, log_prob)
            
            # Update counters
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Render if requested
            if render_training:
                env.render()
            
            if done:
                break
        
        # Update agent after each episode
        agent.update()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Logging
        if episode % 5 == 0:
            avg_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Avg Length: {avg_length:.1f}, Direction: {target_direction}")
            # Inside your environment
            cpg_params = env.current_cpg_action.copy()
            print("Current CPG params:", cpg_params)

        
        # Save model
        if episode % save_interval == 0 and episode > 0:
            model_filename = f"snake_robot_models/ppo_snake_episode_{episode}.pt"
            agent.save(model_filename)
            print(f"Model saved: {model_filename}")
    
    # Final save
    agent.save("snake_robot_models/ppo_snake_final.pt")
    
    # Close environment
    env.close()
    
    return episode_rewards, episode_lengths


def test_snake_robot(model_path="modelV2_with_ground.xml", 
                    checkpoint_path="snake_robot_models/ppo_snake_final.pt",
                    test_direction="forward",
                    test_duration=30):
    """
    Test the trained snake robot
    """
    
    # Create environment with rendering
    env = SnakeRobotEnv(model_path=model_path, render_mode="human")
    
    # Create and load agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim)
    agent.load(checkpoint_path)
    
    print(f"Testing snake robot with direction: {test_direction}")
    
    # Reset environment and record initial position
    state, info = env.reset(options={'target_direction': test_direction})
    initial_position = info.get("position", None)
    
    total_reward = 0
    step_count = 0
    
    while step_count < test_duration * 50:  # Assuming ~50 Hz simulation
        # Select action (deterministic)
        action, _ = agent.select_action(state)
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        state = next_state
        
        # Render
        env.render()
        
        if terminated:
            print("Episode terminated (robot fell or lost stability)")
            break
            
        if truncated:
            print("Episode truncated (time limit)")
            break
    
    final_position = info.get("position", None)
    
    print(f"Test completed. Total reward: {total_reward:.2f}")
    print(f"Initial position: {initial_position}")
    print(f"Final position:   {final_position}")
    
    env.close()


if __name__ == "__main__":
    # Example usage
    print("Snake Robot Reinforcement Learning")
    print("1. Training mode")
    print("2. Testing mode")
    
    mode = input("Select mode (1 or 2): ")
    
    if mode == "1":
        print("Starting training...")
        rewards, lengths = train_snake_robot(
            model_path="modelV2_with_ground.xml",
            num_episodes=2000,  # Reduced for initial testing
            save_interval=100,
            render_training=False  # Set to True if you want to see training
        )
        
        # Save training metrics
        np.save("snake_robot_models/training_rewards.npy", rewards)
        np.save("snake_robot_models/training_lengths.npy", lengths)
        
    elif mode == "2":
        print("Available directions: forward, backward, left, right")
        direction = input("Enter test direction: ")
        
        test_snake_robot(
            model_path="modelV2_with_ground.xml",
            checkpoint_path="snake_robot_models/ppo_snake_final.pt",
            test_direction=direction,
            test_duration=30
        )
    
    else:
        print("Invalid mode selected")