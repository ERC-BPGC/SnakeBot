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

# Parameters
num_joints = 6
amplitude = 0.5
frequency = 1.0
phase_shift = np.pi / 4
offset = 0.0
angle = 0.0

# Create control window
root = tk.Tk()
root.title("CPG Control Panel")
slider_length = 400  # Increased slider length

# Function to create labeled sliders with value display
# Function to create labeled sliders with value display (fixed)
def create_slider(label, from_, to, initial):
    frame = ttk.Frame(root)
    frame.pack(fill='x', padx=5, pady=5)

    ttk.Label(frame, text=label).pack(side='left')

    value_label = ttk.Label(frame, text=f"{initial:.2f}")
    value_label.pack(side='right')

    # Fix: Use a lambda that captures the value_label for each slider
    slider = ttk.Scale(frame, from_=from_, to=to, orient='horizontal', length=slider_length)
    slider.set(initial)
    slider.pack(fill='x')

    # Update label whenever slider moves
    slider.bind("<Motion>", lambda event, lbl=value_label, s=slider: lbl.config(text=f"{float(s.get()):.2f}"))
    slider.bind("<ButtonRelease-1>", lambda event: update_params())  # Update params when released

    return slider


# Sliders
amp_slider = create_slider("Amplitude", 0.0, 1.0, amplitude)
freq_slider = create_slider("Frequency", 0.1, 5.0, frequency)
phase_slider = create_slider("Phase Shift", 0.0, np.pi, phase_shift)
offset_slider = create_slider("Offset", -1.0, 1.0, offset)
angle_slider = create_slider("Angle", -np.pi, np.pi, angle)

# Update parameters
def update_params():
    global amplitude, frequency, phase_shift, offset, angle
    amplitude = float(amp_slider.get())
    frequency = float(freq_slider.get())
    phase_shift = float(phase_slider.get())
    offset = float(offset_slider.get())
    angle = float(angle_slider.get())

# Attach the update function to sliders
for slider in [amp_slider, freq_slider, phase_slider, offset_slider, angle_slider]:
    slider.config(command=lambda x: update_params())
# Function to save parameters to a file
def save_parameters():
    params = {
        "Amplitude": amplitude,
        "Frequency": frequency,
        "Phase Shift": phase_shift,
        "Offset": offset,
        "Angle": angle
    }
    
    # Save the parameters to a text file
    with open("slider_parameters.txt", "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value:.4f}\n")

    print("Parameters saved to slider_parameters.txt")

# Create a save button at the bottom of the window
save_button = ttk.Button(root, text="Save Parameters", command=save_parameters)
save_button.pack(pady=10)

# Access joint limits from the model
joint_limits = model.jnt_range

# Print joint limits for inspection
print("Joint Limits (in radians):")
for i in range(num_joints):
    print(f"Joint {i}: Min = {joint_limits[i][0]:.2f}, Max = {joint_limits[i][1]:.2f}")

# Run MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        root.update_idletasks()
        root.update()

        # CPG Pattern: Apply to vertical and horizontal joints alternately
        for i in range(num_joints):
            control_value   = amplitude * np.sin(2 * np.pi * frequency * t + i * phase_shift) + offset
            data.ctrl[i] = control_value + angle

        mujoco.mj_step(model, data)
        viewer.sync()
        t += 0.01
        time.sleep(0.01)

root.destroy()
