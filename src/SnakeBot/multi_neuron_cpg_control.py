import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import mujoco
import mujoco.viewer
import tkinter as tk
from tkinter import ttk
import threading

# ======================
# Part 1: CPG Integration
# ======================

# Define CPG parameters
n = 5  # Number of CPG neurons
mu = 1.0  # Convergence rate coefficient
theta_tilde = np.pi * np.ones(n-1)  # Desired phase differences
omega = np.ones(n)  # Frequencies of CPG oscillators

# Build Matrix A (convergence velocity matrix)
A = np.zeros((n, n))
for i in range(1, n-1):
    A[i][i-1] = mu
    A[i][i]   = -2 * mu
    A[i][i+1] = mu
A[0][0] = -mu
A[0][1] = mu
A[n-1][n-2] = mu
A[n-1][n-1] = -mu

# Build Matrix B (coupling matrix)
B = np.zeros((n, n-1))
for i in range(n-1):
    B[i][i]   = 1
    B[i+1][i] = -1

# Initial conditions for the phases
phi_init = np.zeros(n)

# Gradient system for the CPG network
def gradient_system(phi, t):
    theta = np.diff(phi)
    dV = np.zeros(n)
    for i in range(1, n-1):
        dV[i] = -2 * mu * (theta[i-1] - theta[i] - (theta_tilde[i-1] - theta_tilde[i]))
    dV[0] = -2 * mu * (theta[0] - theta_tilde[0])
    dV[n-1] = -2 * mu * (theta[n-2] - theta_tilde[n-2])
    
    dphi_dt = omega - np.dot(A, phi) - np.dot(B, theta_tilde)
    return dphi_dt

# Time vector for CPG simulation
time_vec = np.linspace(0, 50, 1000)

# Integrate CPG differential equations
phi_trajectory = odeint(gradient_system, phi_init, time_vec)

# ================================
# Part 2: Sliders and MuJoCo Integration
# ================================

# For simulation, we assume 6-joint model
num_joints = 6

# MuJoCo model loading
model = mujoco.MjModel.from_xml_path("/home/harikrishnan/ROS_PROJECTS/snake_robot_harmonic_ws/src/SnakeBot/scene.xml")
data = mujoco.MjData(model)

# Parameters for control sliders
amplitude_h = 0.5
frequency_h = 0.5
phase_shift_h = np.pi/4
offset_h = 0.0

amplitude_v = 0.3
frequency_v = 0.75
phase_shift_v = np.pi/6
offset_v = 0.0

angle = 0.0

# Create control window using Tkinter
root = tk.Tk()
root.title("CPG Control Panel")
slider_length = 400

# Create labeled sliders with value display function
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
    return slider

# Sliders for horizontal and vertical movement
amp_slider_h = create_slider("Amplitude (Horizontal)", 0.0, 1.0, amplitude_h)
freq_slider_h = create_slider("Frequency (Horizontal)", 0.1, 2.0, frequency_h)
phase_slider_h = create_slider("Phase Shift (Horizontal)", 0.0, np.pi, phase_shift_h)
offset_slider_h = create_slider("Offset (Horizontal)", -1.0, 1.0, offset_h)

amp_slider_v = create_slider("Amplitude (Vertical)", 0.0, 1.0, amplitude_v)
freq_slider_v = create_slider("Frequency (Vertical)", 0.1, 2.0, frequency_v)
phase_slider_v = create_slider("Phase Shift (Vertical)", 0.0, np.pi, phase_shift_v)
offset_slider_v = create_slider("Offset (Vertical)", -1.0, 1.0, offset_v)

# Function to update CPG parameters
def update_params():
    global amplitude_h, frequency_h, phase_shift_h, offset_h
    global amplitude_v, frequency_v, phase_shift_v, offset_v
    
    amplitude_h = float(amp_slider_h.get())
    frequency_h = float(freq_slider_h.get())
    phase_shift_h = float(phase_slider_h.get())
    offset_h = float(offset_slider_h.get())
    
    amplitude_v = float(amp_slider_v.get())
    frequency_v = float(freq_slider_v.get())
    phase_shift_v = float(phase_slider_v.get())
    offset_v = float(offset_slider_v.get())

# Create a save button to store slider values
def save_parameters():
    params = {
        "Amplitude (Horizontal)": amplitude_h,
        "Frequency (Horizontal)": frequency_h,
        "Phase Shift (Horizontal)": phase_shift_h,
        "Offset (Horizontal)": offset_h,
        "Amplitude (Vertical)": amplitude_v,
        "Frequency (Vertical)": frequency_v,
        "Phase Shift (Vertical)": phase_shift_v,
        "Offset (Vertical)": offset_v,
    }
    with open("slider_parameters.txt", "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value:.4f}\n")
    print("Parameters saved to slider_parameters.txt")

# Save button
save_button = ttk.Button(root, text="Save Parameters", command=save_parameters)
save_button.pack(pady=10)

# Update parameters on slider change
for slider in [amp_slider_h, freq_slider_h, phase_slider_h, offset_slider_h,
               amp_slider_v, freq_slider_v, phase_slider_v, offset_slider_v]:
    slider.config(command=lambda x: update_params())

# Function to run MuJoCo in a separate thread
def mujoco_simulation():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_t = 0.0
        dt = 0.01
        while viewer.is_running():
            # Ensure continuous time loop
            sim_t = sim_t % time_vec[-1]  # Loop the simulation time

            # Get the current CPG phases
            idx = np.searchsorted(time_vec, sim_t)
            if idx >= len(time_vec):
                idx = len(time_vec) - 1  # Avoid out-of-bounds error
            cpg_phases = phi_trajectory[idx, :]

            # Control signals for each joint
            for j in range(num_joints):
                phase = cpg_phases[j % n]
                if j % 2 == 0:
                    control_value = amplitude_h * np.sin(2 * np.pi * frequency_h * sim_t + phase + phase_shift_h) + offset_h
                else:
                    control_value = amplitude_v * np.sin(2 * np.pi * frequency_v * sim_t + phase + phase_shift_v) + offset_v
                data.ctrl[j] = control_value + angle

            # Step the simulation forward
            mujoco.mj_step(model, data)
            viewer.sync()
            sim_t += dt  # Increment time
            time.sleep(dt)


# Run MuJoCo in a separate thread
mujoco_thread = threading.Thread(target=mujoco_simulation)
mujoco_thread.start()

# Start Tkinter main loop (runs in the main thread)
root.mainloop()
