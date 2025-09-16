# Snake Robot Training Configuration and Utilities
# Additional utilities and configurations for optimal training

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import json


class TrainingConfig:
    """Configuration class for snake robot training parameters"""
    
    def __init__(self):
        # Environment parameters
        self.max_episode_steps = 1000
        self.reward_weights = {
            'direction': 10.0,      # Reward for moving in target direction
            'smoothness': 0.01,     # Penalty for jerky movements
            'control': 0.001,       # Penalty for high control effort
            'stability': 0.5,       # Penalty for excessive roll/pitch
            'height': 0.1           # Penalty for height deviation
        }
        
        # PPO parameters
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.hidden_dims = [512, 256]
        
        # Training parameters
        self.num_episodes = 5000
        self.save_interval = 100
        self.render_training = False
        
        # GPU settings for RTX 3050 4GB
        self.batch_size = 64      # Smaller batch size for 4GB VRAM
        self.memory_size = 2048   # Limit memory buffer
        self.gradient_clip = 0.5
        
    def save_config(self, filename: str):
        """Save configuration to JSON file"""
        config_dict = {
            'max_episode_steps': self.max_episode_steps,
            'reward_weights': self.reward_weights,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'eps_clip': self.eps_clip,
            'k_epochs': self.k_epochs,
            'hidden_dims': self.hidden_dims,
            'num_episodes': self.num_episodes,
            'save_interval': self.save_interval,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size,
            'gradient_clip': self.gradient_clip
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def load_config(self, filename: str):
        """Load configuration from JSON file"""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class TrainingMetrics:
    """Class to track and visualize training metrics"""
    
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.moving_avg_rewards: List[float] = []
        self.moving_avg_lengths: List[float] = []
        self.direction_success_rates: Dict[str, List[float]] = {
            'forward': [], 'backward': [], 'left': [], 'right': []
        }
    
    def update(self, episode: int, reward: float, length: int, direction: str, success: bool):
        """Update metrics for current episode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Calculate moving averages (last 100 episodes)
        window = min(100, len(self.episode_rewards))
        self.moving_avg_rewards.append(np.mean(self.episode_rewards[-window:]))
        self.moving_avg_lengths.append(np.mean(self.episode_lengths[-window:]))
        
        # Track direction-specific success rates
        if direction in self.direction_success_rates:
            # Calculate success rate for this direction (last 20 episodes)
            direction_episodes = [i for i, d in enumerate(self.get_directions()) if d == direction]
            recent_direction_episodes = [i for i in direction_episodes if i >= len(self.episode_rewards) - 20]
            
            if recent_direction_episodes:
                success_rate = sum(1 for i in recent_direction_episodes if self.is_successful_episode(i)) / len(recent_direction_episodes)
                self.direction_success_rates[direction].append(success_rate)
    
    def get_directions(self) -> List[str]:
        """Get list of directions for each episode (placeholder implementation)"""
        # This would need to be tracked during training
        return ['forward'] * len(self.episode_rewards)
    
    def is_successful_episode(self, episode_idx: int) -> bool:
        """Determine if episode was successful based on reward threshold"""
        return self.episode_rewards[episode_idx] > 50.0  # Adjust threshold as needed
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        ax1.plot(self.moving_avg_rewards, label='Moving Average (100 episodes)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True)
        
        # Episode lengths
        ax2.plot(self.episode_lengths, alpha=0.3, label='Episode Length')
        ax2.plot(self.moving_avg_lengths, label='Moving Average (100 episodes)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        ax2.grid(True)
        
        # Reward distribution
        ax3.hist(self.episode_rewards, bins=50, alpha=0.7)
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reward Distribution')
        ax3.grid(True)
        
        # Success rates by direction
        for direction, success_rates in self.direction_success_rates.items():
            if success_rates:
                ax4.plot(success_rates, label=f'{direction.capitalize()}')
        ax4.set_xlabel('Episode (binned)')
        ax4.set_ylabel('Success Rate')
        ax4.set_title('Success Rate by Direction')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_metrics(self, filename: str):
        """Save metrics to file"""
        metrics_dict = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'moving_avg_rewards': self.moving_avg_rewards,
            'moving_avg_lengths': self.moving_avg_lengths,
            'direction_success_rates': self.direction_success_rates
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_dict, f, indent=4)


class SnakeRobotAnalyzer:
    """Analyze and visualize snake robot performance"""
    
    @staticmethod
    def analyze_model_structure(model_path: str):
        """Analyze the MuJoCo model structure"""
        import mujoco
        
        try:
            model = mujoco.MjModel.from_xml_path(model_path)
            
            print("=== Snake Robot Model Analysis ===")
            print(f"Total bodies: {model.nbody}")
            print(f"Total joints: {model.njnt}")
            print(f"Total actuators: {model.nu}")
            print(f"Total DoF (nq): {model.nq}")
            print(f"Total DoF velocities (nv): {model.nv}")
            
            print("\nJoint Information:")
            for i in range(model.njnt):
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                joint_type = model.jnt_type[i]
                print(f"  Joint {i}: {joint_name}, Type: {joint_type}")
            
            print("\nActuator Information:")
            for i in range(model.nu):
                actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                print(f"  Actuator {i}: {actuator_name}")
            
            return model
            
        except Exception as e:
            print(f"Error analyzing model: {e}")
            return None
    
    @staticmethod
    def visualize_snake_trajectory(positions: np.ndarray, save_path: str = None):
        """Visualize snake robot trajectory"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 2D trajectory
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start')
        ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Snake Robot 2D Trajectory')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # Height over time
        time_steps = np.arange(len(positions))
        ax2.plot(time_steps, positions[:, 2], 'r-', linewidth=2)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Height (m)')
        ax2.set_title('Snake Robot Height Profile')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def setup_gpu_optimization():
    """Setup GPU optimization for RTX 3050 4GB"""
    import torch
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Optimize for 4GB VRAM
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
        
        # Enable mixed precision training
        return True
    else:
        print("CUDA not available, using CPU")
        return False


def create_training_script():
    """Create a simplified training script for easy execution"""
    script_content = '''#!/usr/bin/env python3
"""
Simple training script for Snake Robot RL
Usage: python train_snake.py
"""

from snake_robot_rl import *
from snake_robot_utils import *

def main():
    # Setup configuration
    config = TrainingConfig()
    
    # Optimize for RTX 3050 4GB
    config.batch_size = 32
    config.num_episodes = 2000
    config.learning_rate = 2e-4
    
    # Setup GPU optimization
    gpu_available = setup_gpu_optimization()
    
    print("Starting Snake Robot Training...")
    print(f"GPU Available: {gpu_available}")
    print(f"Episodes: {config.num_episodes}")
    
    # Train the robot
    rewards, lengths = train_snake_robot(
        model_path="modelV2_with_ground.xml",
        num_episodes=config.num_episodes,
        save_interval=config.save_interval,
        render_training=False
    )
    
    # Create metrics and visualize
    metrics = TrainingMetrics()
    metrics.episode_rewards = rewards
    metrics.episode_lengths = lengths
    
    # Calculate moving averages
    for i in range(len(rewards)):
        window = min(100, i + 1)
        metrics.moving_avg_rewards.append(np.mean(rewards[max(0, i-window+1):i+1]))
        metrics.moving_avg_lengths.append(np.mean(lengths[max(0, i-window+1):i+1]))
    
    # Save results
    metrics.save_metrics("training_metrics.json")
    metrics.plot_training_progress("training_progress.png")
    
    print("Training completed!")
    print(f"Final average reward: {np.mean(rewards[-100:]):.2f}")

if __name__ == "__main__":
    main()
'''
    
    with open("train_snake.py", "w") as f:
        f.write(script_content)
    
    print("Created train_snake.py - Simple training script")


def create_test_script():
    """Create a testing script for the trained model"""
    script_content = '''#!/usr/bin/env python3
"""
Test script for trained Snake Robot
Usage: python test_snake.py [direction] [duration]
"""

import sys
from snake_robot_rl import test_snake_robot

def main():
    # Parse command line arguments
    direction = sys.argv[1] if len(sys.argv) > 1 else "forward"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    # Available directions
    valid_directions = ["forward", "backward", "left", "right"]
    
    if direction not in valid_directions:
        print(f"Invalid direction. Choose from: {valid_directions}")
        return
    
    print(f"Testing snake robot - Direction: {direction}, Duration: {duration}s")
    
    test_snake_robot(
        model_path="modelV2_with_ground.xml",
        checkpoint_path="snake_robot_models/ppo_snake_final.pt",
        test_direction=direction,
        test_duration=duration
    )

if __name__ == "__main__":
    main()
'''
    
    with open("test_snake.py", "w") as f:
        f.write(script_content)
    
    print("Created test_snake.py - Simple testing script")


if __name__ == "__main__":
    # Create utility scripts
    create_training_script()
    create_test_script()
    
    # Analyze the provided model
    print("Analyzing your MuJoCo model...")
    model = SnakeRobotAnalyzer.analyze_model_structure("/home/harikrishnan/ROS_PROJECTS/snake_robot_harmonic_ws/snake_robot_harmonic_ws/src/SnakeBot/Gymnsium_SIM/my_env/robot_model/modelV2_with_ground.xml")
    
    # Setup GPU optimization
    setup_gpu_optimization()