#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class JointCommander(Node):
    def __init__(self):
        super().__init__('joint_commander')

        # List of joint topics
        joint_topics = [
            '/joint_state1', '/joint_state2', '/joint_state3', '/joint_state4',
            '/joint_state5', '/joint_state6', '/joint_state7', '/joint_state8',
            '/joint_state9', '/joint_state10', '/joint_state11', '/joint_state12'
        ]

        # Create publishers for each joint
        self.publishers = {topic: self.create_publisher(Float64, topic, 10) for topic in joint_topics}

        # Set a timer to publish commands periodically
        self.timer = self.create_timer(1.0, self.publish_commands)

        # Command values for each joint
        self.commands = [0.5, -0.5, 0.3, -0.3, 1.0, -1.0, 0.8, -0.8, 0.0, 0.2, -0.2, 0.1]

    def publish_commands(self):
        for i, (topic, publisher) in enumerate(self.publishers.items()):
            msg = Float64()
            msg.data = self.commands[i]
            publisher.publish(msg)
            self.get_logger().info(f'Published {msg.data} to {topic}')

def main(args=None):
    rclpy.init(args=args)
    node = JointCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
