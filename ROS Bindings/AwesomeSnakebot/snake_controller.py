#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import time

class SimpleSnakeTestDebug(Node):
    def __init__(self):
        super().__init__('simple_snake_test_debug')
        self.get_logger().info('Initializing SimpleSnakeTestDebug Node...')
        
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        
        # Define parameters for angle calculation
        self.num_segments = 6
        self.frequency = 1
        self.speed = 25
        self.get_logger().info(f'Parameters: num_segments={self.num_segments}, frequency={self.frequency}, speed={self.speed}')
        
        self.t_values = np.linspace(0, 2 * np.pi, 100)  # Simulating t values like in the Arduino code
        self.t_index = 0  # To keep track of the current t value
        
        # Initialize list to hold joint angles
        self.delay=0.5
        self.joint_angles = [0.0] * self.num_segments
        self.get_logger().info('Node initialized successfully.')

    def calculate_angles(self, t):
        t = t * self.speed
        self.get_logger().info(f'Calculating angles for t={t}')
        angles = []
        for i in range(self.num_segments):
            angle = np.sin(2 * np.pi * self.frequency * t + i * np.pi / self.num_segments)
            angles.append(angle)
            self.get_logger().debug(f'Joint {i + 1}: angle={angle}')
        return angles

    def calculate_segment_angles(self, angles):
        self.get_logger().info(f'Calculating segment angles from joint angles: {angles}')
        segment_angles = [angles[i + 1] - angles[i] for i in range(self.num_segments - 1)]
        self.get_logger().debug(f'Segment angles: {segment_angles}')
        return segment_angles

    def run(self):
        self.get_logger().info('Starting main loop...')
        while rclpy.ok():
            # Get the current t value
            t = self.t_values[self.t_index]
            self.get_logger().info(f'Current t value: {t} (index {self.t_index})')
            self.t_index = (self.t_index + 1) % len(self.t_values)  # Loop through t values

            # Calculate joint angles
            angles = self.calculate_angles(t)
            segment_angles = self.calculate_segment_angles(angles)
            
            # Create JointState message
            joint_state = JointState()
            joint_state.header.stamp = self.get_clock().now().to_msg()
            joint_state.name = [f'joint{i + 1}' for i in range(self.num_segments)]
            joint_state.position = angles  # Publish angles

            # Publish the joint state
            self.publisher_.publish(joint_state)
            self.get_logger().info(f'Published JointState: {joint_state.position}')
            self.get_logger().debug(f'Joint Names: {joint_state.name}')
            self.get_logger().debug(f'Timestamp: {joint_state.header.stamp}')

            # Sleep for 0.1 seconds
            self.get_logger().info('Sleeping for 0.1 seconds...')
            time.sleep(self.delay)
        self.get_logger().info('Exiting main loop.')

def main(args=None):
    rclpy.init(args=args)
    simple_snake_test_debug = SimpleSnakeTestDebug()
    try:
        simple_snake_test_debug.run()
    except KeyboardInterrupt:
        simple_snake_test_debug.get_logger().info('KeyboardInterrupt received, shutting down...')
    finally:
        simple_snake_test_debug.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
