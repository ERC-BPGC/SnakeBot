#!/usr/bin/env python3
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
        model_path="/home/harikrishnan/ROS_PROJECTS/snake_robot_harmonic_ws/snake_robot_harmonic_ws/src/SnakeBot/Gymnsium_SIM/my_env/robot_model/modelV2_with_ground.xml",
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
