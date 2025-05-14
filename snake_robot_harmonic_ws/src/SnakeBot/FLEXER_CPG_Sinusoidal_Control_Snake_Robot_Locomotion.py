import mujoco
import mujoco.viewer
import numpy as np
import time
import tkinter as tk
from tkinter import ttk

# Path to the MJCF file
MJCF_PATH = "/home/harikrishnan/ROS_PROJECTS/snake_robot_harmonic_ws/src/SnakeBot/scene.xml"

# Load the model
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)

# Parameters for sinusoidal wave (for CPG control)
num_joints = 14 # Total number of joints in the snake robot
amplitude = 0.5
frequency = 1.0
phase_shift = np.pi / 4
offset = 0.0
angle = 0.0

# Central Pattern Generator (CPG) Parameters
k_n = 2.0  # Wave number, can be adjusted for number of waves
L = 1.0  # Length of the snake body (scaled for simplicity)
t_r = 0.2  # Time constant for the response of the neurons
t_a = 0.2  # Time constant for the output of the neurons
w_ef = 1.0  # Connection strength between extensor and flexor neurons
w_ij = 1.0  # Connection strengths between extensor neurons
u_0 = 0.0  # Constant input to the system
F_i = 0.0  # External force or influence
b = 0.1  # Damping coefficient

# Create control window using Tkinter for slider control
root = tk.Tk()
root.title("CPG Control Panel")
slider_length = 400

def create_slider(label, from_, to, initial):
    frame = ttk.Frame(root)
    frame.pack(fill='x', padx=5, pady=5)

    ttk.Label(frame, text=label).pack(side='left')

    value_label = ttk.Label(frame, text=f"{initial:.2f}")
    value_label.pack(side='right')

    slider = ttk.Scale(frame, from_=from_, to=to, orient='horizontal', length=slider_length)
    slider.set(initial)
    slider.pack(fill='x')

    slider.bind("<Motion>", lambda event, lbl=value_label, s=slider: lbl.config(text=f"{float(s.get()):.2f}"))
    slider.bind("<ButtonRelease-1>", lambda event: update_params())  # Update params when released

    return slider

# Sliders for adjusting CPG parameters
amp_slider = create_slider("Amplitude", 0.0, 1.0, amplitude)
freq_slider = create_slider("Frequency", 0.1, 5.0, frequency)
phase_slider = create_slider("Phase Shift", 0.0, np.pi, phase_shift)
offset_slider = create_slider("Offset", -1.0, 1.0, offset)
angle_slider = create_slider("Angle", -np.pi, np.pi, angle)

# Update parameters when slider values change
def update_params():
    global amplitude, frequency, phase_shift, offset, angle, k_n, L
    amplitude = float(amp_slider.get())
    frequency = float(freq_slider.get())
    phase_shift = float(phase_slider.get())
    offset = float(offset_slider.get())
    angle = float(angle_slider.get())

# Function to calculate the CPG signals for each joint
def calculate_cpg_signal(t, joint_index):
    """Calculate the control signal based on the CPG model for each joint."""
    # Sinusoidal wave equation (curvature)
    y = amplitude * np.cos(2 * np.pi * frequency * t + joint_index * phase_shift)
    # Apply wave number and body length for spatial distribution
    control_value = y + offset + angle  # Adding offset and angle for adjustment
    return control_value

# Create save button for parameters
def save_parameters():
    params = {
        "Amplitude": amplitude,
        "Frequency": frequency,
        "Phase Shift": phase_shift,
        "Offset": offset,
        "Angle": angle
    }
    
    with open("slider_parameters.txt", "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value:.4f}\n")

    print("Parameters saved to slider_parameters.txt")

save_button = ttk.Button(root, text="Save Parameters", command=save_parameters)
save_button.pack(pady=10)

# Access joint limits from the model (optional)
joint_limits = model.jnt_range
print("Joint Limits (in radians):")
for i in range(num_joints):
    print(f"Joint {i}: Min = {joint_limits[i][0]:.2f}, Max = {joint_limits[i][1]:.2f}")

# Run the MuJoCo viewer and control the snake robot using CPG-based signals
with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        root.update_idletasks()
        root.update()

        # Apply CPG control signals to each joint
        for i in range(num_joints):
            control_value = calculate_cpg_signal(t, i)
            data.ctrl[i] = control_value  # Set control value for each joint

        # Advance the simulation step
        mujoco.mj_step(model, data)
        viewer.sync()

        t += 0.01  # Increment time step
        time.sleep(0.01)  # Delay for smooth simulation

root.destroy()
