#!/usr/bin/env python3
import rclpy
import math
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64

class CPGController(Node):
    def __init__(self):
        super().__init__('cpg_controller')

        # Number of joints
        self.n = 12  # 12-joint snake robot

        # ROS 2 publishers for each joint
        self.joint_publishers = [self.create_publisher(Float64, f'/joint_state{i+1}', 10) for i in range(self.n)]

        # Timer for update loop
        self.timer = self.create_timer(0.1, self.update_cpg)  # 10 Hz update

        # Phase and amplitude state variables
        self.phi = np.zeros(self.n)  # Phase variables
        self.r = np.ones(self.n) * 30  # Amplitudes (degrees)
        self.r_dot = np.zeros(self.n)  # Amplitude derivatives

        # Desired Amplitude (target for r)
        self.R = np.ones(self.n) * 30  # Target amplitude

        # Phase frequency
        self.omega = np.ones(self.n) * 1.5  # Hz

        # Phase difference matrix (for smooth transitions)
        self.A = self.build_phase_coupling_matrix()

        # Phase shift between joints (wave propagation)
        self.theta = np.ones(self.n - 1) * (math.pi / 6)

        # Offset
        self.delta = np.zeros(self.n)

        # Convergence rate for amplitude adaptation
        self.a = 1.0  

        # Time tracking
        self.time = 0.0  

    def build_phase_coupling_matrix(self):
        """ Builds an n × n phase coupling matrix A """
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            if i > 0:
                A[i, i - 1] = 1
            if i < self.n - 1:
                A[i, i + 1] = 1
            A[i, i] = -2 if i > 0 and i < self.n - 1 else -1
        return A

    def update_cpg(self):
        """ Updates CPG state and publishes joint angles to ROS 2 """
        
        # Compute phase update (gradient descent method)
        phi_dot = self.omega + np.dot(self.A, self.phi) + np.dot(self.B_matrix(), self.theta)
        self.phi += phi_dot * 0.1  # Euler integration

        # Compute amplitude update
        r_ddot = self.a * ((self.a / 4) * (self.R - self.r) - self.r_dot)
        self.r_dot += r_ddot * 0.1  # Euler integration
        self.r += self.r_dot * 0.1

        # Compute final joint angles
        joint_angles = self.r * np.sin(self.phi) + self.delta

        # Publish joint commands
        for i in range(self.n):
            msg = Float64()
            msg.data = math.radians(joint_angles[i])  # Convert degrees to radians
            self.joint_publishers[i].publish(msg)

        # Update time
        self.time += 0.1

    def B_matrix(self):
        """ Builds an n × (n-1) matrix B for phase shift propagation """
        B = np.zeros((self.n, self.n - 1))
        for i in range(self.n - 1):
            B[i, i] = 1
            B[i + 1, i] = -1
        return B

def main(args=None):
    rclpy.init(args=args)
    node = CPGController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
