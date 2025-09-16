#!/usr/bin/env python3
"""
Test script for trained Snake Robot
Usage: python test_snake.py [direction] [duration]
"""

import sys
from snake_robot_rl import test_snake_robot

def main():
    # Parse command line arguments
    direction = "forward"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    # Available directions
    valid_directions = ["forward", "backward", "left", "right"]
    
    if direction not in valid_directions:
        print(f"Invalid direction. Choose from: {valid_directions}")
        return
    
    print(f"Testing snake robot - Direction: {direction}, Duration: {duration}s")
    
    test_snake_robot(
        model_path="/home/harikrishnan/ROS_PROJECTS/snake_robot_harmonic_ws/snake_robot_harmonic_ws/src/SnakeBot/Gymnsium_SIM/my_env/robot_model/modelV2_with_ground.xml",
        checkpoint_path="snake_robot_models/ppo_snake_episode_200.pt",
        test_direction=direction,
        test_duration=duration
    )

if __name__ == "__main__":
    main()
