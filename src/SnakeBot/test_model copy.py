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

# Update parameters
def update_params():
    global amplitude, frequency, phase_shift, offset, angle
    amplitude = float(amp_slider.get())
    frequency = float(freq_slider.get())
    phase_shift = float(phase_slider.get())
    offset = float(offset_slider.get())
    angle = float(angle_slider.get())

# Create control window
root = tk.Tk()
root.title("CPG Control Panel")

slider_length = 400  # Increased slider length

# Sliders
ttk.Label(root, text="Amplitude").pack()
amp_slider = ttk.Scale(root, from_=0.0, to=1.0, orient='horizontal', length=slider_length, command=lambda x: update_params())
amp_slider.set(amplitude)
amp_slider.pack()

ttk.Label(root, text="Frequency").pack()
freq_slider = ttk.Scale(root, from_=0.1, to=5.0, orient='horizontal', length=slider_length, command=lambda x: update_params())
freq_slider.set(frequency)
freq_slider.pack()

ttk.Label(root, text="Phase Shift").pack()
phase_slider = ttk.Scale(root, from_=0.0, to=np.pi, orient='horizontal', length=slider_length, command=lambda x: update_params())
phase_slider.set(phase_shift)
phase_slider.pack()

ttk.Label(root, text="Offset").pack()
offset_slider = ttk.Scale(root, from_=-1.0, to=1.0, orient='horizontal', length=slider_length, command=lambda x: update_params())
offset_slider.set(offset)
offset_slider.pack()

ttk.Label(root, text="Angle").pack()
angle_slider = ttk.Scale(root, from_=-np.pi, to=np.pi, orient='horizontal', length=slider_length, command=lambda x: update_params())
angle_slider.set(angle)
angle_slider.pack()

# Run MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        root.update_idletasks()
        root.update()

        # CPG Pattern: Apply to vertical and horizontal joints alternately
        for i in range(num_joints):
            control_value = amplitude * np.sin(2 * np.pi * frequency * t + i * phase_shift) + offset
            data.ctrl[i] = control_value + angle

        mujoco.mj_step(model, data)
        viewer.sync()
        t += 0.01
        time.sleep(0.01)

root.destroy()