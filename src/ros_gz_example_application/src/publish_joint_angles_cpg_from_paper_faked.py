#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node
from std_msgs.msg import Float64

from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QGroupBox
from PyQt5.QtCore import Qt, QTimer
import numpy as np

class SnakeMovementController(Node):
    def __init__(self):
        super().__init__('snake_movement_controller')
        self.n = 12  # total joints

        # Create publishers for each joint
        self.joint_publishers = []
        for i in range(1, self.n + 1):
            topic_name = f'/joint_state{i}'
            publisher = self.create_publisher(Float64, topic_name, 10)
            self.joint_publishers.append(publisher)

        # Divide joints into two groups: odd and even
        self.n_odd = self.n // 2   # joints 1,3,5,... (indexes 0,2,4,...)
        self.n_even = self.n // 2  # joints 2,4,6,... (indexes 1,3,5,...)

        # Integration step and common dynamics parameters
        self.dt = 0.1  # seconds
        self.a = 1.0   # parameter for amplitude dynamics
        self.mu = 1.0  # hyperparameter for coupling

        # ---------------------------
        # Odd Joint CPG States & Matrices
        # ---------------------------
        self.phi_odd = np.zeros(self.n_odd)          # phase states for odd joints
        self.r_odd = np.ones(self.n_odd)               # amplitude states
        self.r_dot_odd = np.zeros(self.n_odd)          # amplitude derivatives

        self.A_odd = np.zeros((self.n_odd, self.n_odd))
        for i in range(self.n_odd):
            if i == 0:
                self.A_odd[0, 0] = -self.mu
                if self.n_odd > 1:
                    self.A_odd[0, 1] = self.mu
            elif i == self.n_odd - 1:
                self.A_odd[i, i-1] = self.mu
                self.A_odd[i, i] = -self.mu
            else:
                self.A_odd[i, i-1] = self.mu
                self.A_odd[i, i] = -2 * self.mu
                self.A_odd[i, i+1] = self.mu

        if self.n_odd > 1:
            self.B_odd = np.zeros((self.n_odd, self.n_odd - 1))
            self.B_odd[0, 0] = 1
            for i in range(1, self.n_odd - 1):
                self.B_odd[i, i-1] = -1
                self.B_odd[i, i] = 1
            self.B_odd[self.n_odd - 1, self.n_odd - 2] = -1
        else:
            self.B_odd = np.array([])

        # ---------------------------
        # Even Joint CPG States & Matrices
        # ---------------------------
        self.phi_even = np.zeros(self.n_even)
        self.r_even = np.ones(self.n_even)
        self.r_dot_even = np.zeros(self.n_even)

        self.A_even = np.zeros((self.n_even, self.n_even))
        for i in range(self.n_even):
            if i == 0:
                self.A_even[0, 0] = -self.mu
                if self.n_even > 1:
                    self.A_even[0, 1] = self.mu
            elif i == self.n_even - 1:
                self.A_even[i, i-1] = self.mu
                self.A_even[i, i] = -self.mu
            else:
                self.A_even[i, i-1] = self.mu
                self.A_even[i, i] = -2 * self.mu
                self.A_even[i, i+1] = self.mu

        if self.n_even > 1:
            self.B_even = np.zeros((self.n_even, self.n_even - 1))
            self.B_even[0, 0] = 1
            for i in range(1, self.n_even - 1):
                self.B_even[i, i-1] = -1
                self.B_even[i, i] = 1
            self.B_even[self.n_even - 1, self.n_even - 2] = -1
        else:
            self.B_even = np.array([])

        # ---------------------------
        # Separate CPG Input Parameters for Odd and Even Networks
        # ---------------------------
        # Odd group parameters
        self.amplitude_deg_odd = 30   # amplitude in degrees
        self.frequency_odd = 0.5      # frequency in Hz
        self.phase_deg_odd = 30       # phase shift in degrees
        self.offset_deg_odd = 0       # offset in degrees

        self.R_odd = np.full(self.n_odd, math.radians(self.amplitude_deg_odd))
        self.omega_odd = np.full(self.n_odd, 2 * math.pi * self.frequency_odd)
        if self.n_odd > 1:
            self.theta_odd = np.full(self.n_odd - 1, math.radians(self.phase_deg_odd))
        else:
            self.theta_odd = np.array([])
        self.delta_odd = np.full(self.n_odd, math.radians(self.offset_deg_odd))

        # Even group parameters
        self.amplitude_deg_even = 30
        self.frequency_even = 0.5
        self.phase_deg_even = 30
        self.offset_deg_even = 0

        self.R_even = np.full(self.n_even, math.radians(self.amplitude_deg_even))
        self.omega_even = np.full(self.n_even, 2 * math.pi * self.frequency_even)
        if self.n_even > 1:
            self.theta_even = np.full(self.n_even - 1, math.radians(self.phase_deg_even))
        else:
            self.theta_even = np.array([])
        self.delta_even = np.full(self.n_even, math.radians(self.offset_deg_even))

        # For publishing: store the combined joint outputs (length = n)
        self.joint_angles = [0] * self.n

        # Timer to update the CPG networks
        self.timer = self.create_timer(self.dt, self.update_joint_angles)

    def update_joint_angles(self):
        # --- Odd Group Update ---
        if self.n_odd > 1:
            phi_dot_odd = self.omega_odd + self.A_odd @ self.phi_odd + self.B_odd @ self.theta_odd
        else:
            phi_dot_odd = self.omega_odd
        self.phi_odd = self.phi_odd + phi_dot_odd * self.dt

        r_ddot_odd = self.a * ((self.a / 4) * (self.R_odd - self.r_odd) - self.r_dot_odd)
        self.r_dot_odd = self.r_dot_odd + r_ddot_odd * self.dt
        self.r_odd = self.r_odd + self.r_dot_odd * self.dt

        x_odd = self.r_odd * np.sin(self.phi_odd) + self.delta_odd

        # --- Even Group Update ---
        if self.n_even > 1:
            phi_dot_even = self.omega_even + self.A_even @ self.phi_even + self.B_even @ self.theta_even
        else:
            phi_dot_even = self.omega_even
        self.phi_even = self.phi_even + phi_dot_even * self.dt

        r_ddot_even = self.a * ((self.a / 4) * (self.R_even - self.r_even) - self.r_dot_even)
        self.r_dot_even = self.r_dot_even + r_ddot_even * self.dt
        self.r_even = self.r_even + self.r_dot_even * self.dt

        x_even = self.r_even * np.sin(self.phi_even) + self.delta_even

        # --- Combine Outputs ---
        x = np.zeros(self.n)
        x[0::2] = x_odd   # odd joints (indexes 0,2,4,…)
        x[1::2] = x_even  # even joints (indexes 1,3,5,…)

        # Publish joint angles
        for i in range(self.n):
            angle_rad = x[i]
            self.joint_angles[i] = angle_rad
            msg = Float64()
            msg.data = angle_rad
            self.joint_publishers[i].publish(msg)

    # --- Parameter update methods for the Odd Group ---
    def update_amplitude_odd(self, value):
        self.amplitude_deg_odd = value
        self.R_odd = np.full(self.n_odd, math.radians(value))
    
    def update_frequency_odd(self, value):
        self.frequency_odd = value / 10.0
        self.omega_odd = np.full(self.n_odd, 2 * math.pi * self.frequency_odd)
    
    def update_phase_shift_odd(self, value):
        self.phase_deg_odd = value
        if self.n_odd > 1:
            self.theta_odd = np.full(self.n_odd - 1, math.radians(value))
    
    def update_offset_odd(self, value):
        self.offset_deg_odd = value
        self.delta_odd = np.full(self.n_odd, math.radians(value))

    # --- Parameter update methods for the Even Group ---
    def update_amplitude_even(self, value):
        self.amplitude_deg_even = value
        self.R_even = np.full(self.n_even, math.radians(value))
    
    def update_frequency_even(self, value):
        self.frequency_even = value / 10.0
        self.omega_even = np.full(self.n_even, 2 * math.pi * self.frequency_even)
    
    def update_phase_shift_even(self, value):
        self.phase_deg_even = value
        if self.n_even > 1:
            self.theta_even = np.full(self.n_even - 1, math.radians(value))
    
    def update_offset_even(self, value):
        self.offset_deg_even = value
        self.delta_even = np.full(self.n_even, math.radians(value))


# ---------------------------------------------------
# GUI Class: Separate Controls for Odd and Even Groups
# ---------------------------------------------------
class ControlGUI(QMainWindow):
    def __init__(self, movement_controller):
        super().__init__()
        self.controller = movement_controller
        self.setWindowTitle("Snake Robot Dual CPG Control Panel - Separate Controls")
        
        # Create group boxes for Odd and Even controls
        odd_group_box = QGroupBox("Odd Joint CPG Controls")
        even_group_box = QGroupBox("Even Joint CPG Controls")
        
        # Odd group sliders
        odd_layout = QVBoxLayout()
        self.odd_amp_slider = self.create_slider("Amplitude (deg)", 0, 90, self.controller.amplitude_deg_odd, self.update_amplitude_odd)
        self.odd_freq_slider = self.create_slider("Frequency (Hz x10)", 1, 20, int(self.controller.frequency_odd * 10), self.update_frequency_odd)
        self.odd_phase_slider = self.create_slider("Phase Shift (deg)", 0, 360, int(self.controller.phase_deg_odd), self.update_phase_shift_odd)
        self.odd_offset_slider = self.create_slider("Offset (deg)", -90, 90, self.controller.offset_deg_odd, self.update_offset_odd)
        odd_layout.addWidget(self.odd_amp_slider["label"])
        odd_layout.addWidget(self.odd_amp_slider["slider"])
        odd_layout.addWidget(self.odd_freq_slider["label"])
        odd_layout.addWidget(self.odd_freq_slider["slider"])
        odd_layout.addWidget(self.odd_phase_slider["label"])
        odd_layout.addWidget(self.odd_phase_slider["slider"])
        odd_layout.addWidget(self.odd_offset_slider["label"])
        odd_layout.addWidget(self.odd_offset_slider["slider"])
        odd_group_box.setLayout(odd_layout)
        
        # Even group sliders
        even_layout = QVBoxLayout()
        self.even_amp_slider = self.create_slider("Amplitude (deg)", 0, 90, self.controller.amplitude_deg_even, self.update_amplitude_even)
        self.even_freq_slider = self.create_slider("Frequency (Hz x10)", 1, 20, int(self.controller.frequency_even * 10), self.update_frequency_even)
        self.even_phase_slider = self.create_slider("Phase Shift (deg)", 0, 360, int(self.controller.phase_deg_even), self.update_phase_shift_even)
        self.even_offset_slider = self.create_slider("Offset (deg)", -90, 90, self.controller.offset_deg_even, self.update_offset_even)
        even_layout.addWidget(self.even_amp_slider["label"])
        even_layout.addWidget(self.even_amp_slider["slider"])
        even_layout.addWidget(self.even_freq_slider["label"])
        even_layout.addWidget(self.even_freq_slider["slider"])
        even_layout.addWidget(self.even_phase_slider["label"])
        even_layout.addWidget(self.even_phase_slider["slider"])
        even_layout.addWidget(self.even_offset_slider["label"])
        even_layout.addWidget(self.even_offset_slider["slider"])
        even_group_box.setLayout(even_layout)
        
        # Combine the two group boxes in a horizontal layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(odd_group_box)
        main_layout.addWidget(even_group_box)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
    
    def create_slider(self, name, min_val, max_val, init_val, callback):
        label = QLabel(f"{name}: {init_val}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.valueChanged.connect(lambda value: callback(value, label))
        return {"label": label, "slider": slider}
    
    # --- Update methods for the Odd Group ---
    def update_amplitude_odd(self, value, label):
        self.controller.update_amplitude_odd(value)
        label.setText(f"Amplitude (deg): {value}")
    
    def update_frequency_odd(self, value, label):
        self.controller.update_frequency_odd(value)
        label.setText(f"Frequency: {self.controller.frequency_odd:.1f} Hz")
    
    def update_phase_shift_odd(self, value, label):
        self.controller.update_phase_shift_odd(value)
        label.setText(f"Phase Shift (deg): {value}")
    
    def update_offset_odd(self, value, label):
        self.controller.update_offset_odd(value)
        label.setText(f"Offset (deg): {value}")
    
    # --- Update methods for the Even Group ---
    def update_amplitude_even(self, value, label):
        self.controller.update_amplitude_even(value)
        label.setText(f"Amplitude (deg): {value}")
    
    def update_frequency_even(self, value, label):
        self.controller.update_frequency_even(value)
        label.setText(f"Frequency: {self.controller.frequency_even:.1f} Hz")
    
    def update_phase_shift_even(self, value, label):
        self.controller.update_phase_shift_even(value)
        label.setText(f"Phase Shift (deg): {value}")
    
    def update_offset_even(self, value, label):
        self.controller.update_offset_even(value)
        label.setText(f"Offset (deg): {value}")

def main(args=None):
    rclpy.init(args=args)
    movement_controller = SnakeMovementController()
    
    # Start the GUI
    app = QApplication([])
    gui = ControlGUI(movement_controller)
    gui.show()
    
    # Run ROS spinning in a separate thread
    import threading
    ros_thread = threading.Thread(target=rclpy.spin, args=(movement_controller,), daemon=True)
    ros_thread.start()
    
    app.exec()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
