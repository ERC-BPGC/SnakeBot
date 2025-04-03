#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from std_msgs.msg import Float64
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class SnakeMovementController(Node):

    def __init__(self):
        super().__init__('snake_movement_controller')

        # Define publishers for each joint (1 to 12)
        self.joint_publishers = []
        for i in range(1, 13):  # 12 joints
            topic_name = f'/joint_state{i}'
            publisher = self.create_publisher(Float64, topic_name, 10)
            self.joint_publishers.append(publisher)

        # Initialize parameters for CPG
        self.amplitude = 30  # Amplitude of joint angle in degrees
        self.frequency = 0.5  # Frequency of oscillation in Hz
        self.phase_shift = math.pi / 6  # Phase shift between joints
        self.time = 0  # Time variable for the sinusoidal wave

        # Initialize timers to call the movement function
        self.timer = self.create_timer(0.1, self.update_joint_angles)  # 10Hz update rate

        # Store joint angles for plotting
        self.joint_angles = [0] * 12

    def update_joint_angles(self):
        """
        Update joint angles based on a CPG model (sinusoidal functions).
        This will simulate a wave-like motion for the snake robot.
        """
        for joint_num in range(12):  # Loop through 12 joints
            # Calculate the phase shift for each joint
            phase = joint_num * self.phase_shift

            # Calculate the angle using a sinusoidal function for the CPG
            angle = self.amplitude * math.sin(2 * math.pi * self.frequency * self.time + phase)

            # Keep the joint angle within the range (-180 to 180 degrees)
            angle = max(min(angle, 180), -180)

            # Convert the angle from degrees to radians
            angle_rad = math.radians(angle)

            # Update joint angle for plotting
            self.joint_angles[joint_num] = angle_rad

            # Set the joint's angle for movement control
            joint_state_msg = Float64()
            joint_state_msg.data = angle_rad  # Control robot movement based on the angle in radians
            self.joint_publishers[joint_num].publish(joint_state_msg)

        # Increment time for the next wave cycle
        self.time += 0.1  # Increment time by 0.1 seconds for the next loop


class ControlGUI(QMainWindow):
    def __init__(self, movement_controller):
        super().__init__()
        self.controller = movement_controller
        self.setWindowTitle("Snake Robot Control Panel")

        # Create sliders for amplitude, frequency, and phase shift
        self.amplitude_slider = self.create_slider("Amplitude", 0, 300, self.controller.amplitude, self.update_amplitude)
        self.frequency_slider = self.create_slider("Frequency", 1, 20, int(self.controller.frequency * 10), self.update_frequency)
        self.phase_slider = self.create_slider("Phase Shift", 0, 360, int(math.degrees(self.controller.phase_shift)), self.update_phase_shift)

        # Create a layout for sliders
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.amplitude_slider["label"])
        slider_layout.addWidget(self.amplitude_slider["slider"])
        slider_layout.addWidget(self.frequency_slider["label"])
        slider_layout.addWidget(self.frequency_slider["slider"])
        slider_layout.addWidget(self.phase_slider["label"])
        slider_layout.addWidget(self.phase_slider["slider"])

        # Add plotting canvas
        self.plot_canvas = PlotCanvas(self.controller)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.plot_canvas)

        # Combine sliders and plot in a horizontal layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(slider_layout)
        main_layout.addLayout(plot_layout)

        # Set the central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Update the plot regularly
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.plot_canvas.update_plot)
        self.plot_timer.start(100)  # Update every 100 ms

    def create_slider(self, name, min_val, max_val, init_val, callback):
        label = QLabel(f"{name}: {init_val}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.valueChanged.connect(lambda value: callback(value, label))
        return {"label": label, "slider": slider}

    def update_amplitude(self, value, label):
        self.controller.amplitude = value
        label.setText(f"Amplitude: {value}")

    def update_frequency(self, value, label):
        self.controller.frequency = value / 10.0
        label.setText(f"Frequency: {self.controller.frequency:.1f}")

    def update_phase_shift(self, value, label):
        self.controller.phase_shift = math.radians(value)
        label.setText(f"Phase Shift: {value}°")


class PlotCanvas(FigureCanvas):
    def __init__(self, controller):
        self.controller = controller
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.time_data = np.linspace(0, 1, 12)  # X-axis for joint numbers
        self.angle_data = [0] * 12  # Y-axis for joint angles

        # Initial plot
        self.line, = self.ax.plot(self.time_data, self.angle_data, marker='o', label='Joint Angles')
        self.ax.set_title("Joint Angles")
        self.ax.set_xlabel("Joint Number")
        self.ax.set_ylabel("Angle (radians)")
        self.ax.set_ylim(-math.pi, math.pi)
        self.ax.legend()

    def update_plot(self):
        self.angle_data = self.controller.joint_angles
        self.line.set_ydata(self.angle_data)
        self.fig.canvas.draw()


def main(args=None):
    rclpy.init(args=args)
    movement_controller = SnakeMovementController()

    # Start the GUI
    app = QApplication([])
    gui = ControlGUI(movement_controller)
    gui.show()

    # Spin ROS node in a separate thread
    import threading
    ros_thread = threading.Thread(target=rclpy.spin, args=(movement_controller,), daemon=True)
    ros_thread.start()

    app.exec()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
