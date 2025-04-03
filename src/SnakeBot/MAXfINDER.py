import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

# Path to the MJCF file (scene.xml includes n2.xml)
MJCF_PATH = "/home/harikrishnan/ROS_PROJECTS/snake_robot_harmonic_ws/src/SnakeBot/scene.xml"

# Load the model and simulation data
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)

# Determine the number of actuators (and assume one actuator per joint)
num_actuators = model.nu

# Read the absolute control limits from the model (for each actuator)
print("Absolute actuator control limits (from ctrlrange):")
print(model.actuator_ctrlrange)

# We'll record the joint angle history (in degrees) for a particular actuator.
# For this example, we'll test the first actuator (index 0).
joint_history = []
t = 0.0

# Ramp settings: We'll sweep from the minimum to the maximum value of the actuator.
ctrl_min, ctrl_max = model.actuator_ctrlrange[0]
step_size = (ctrl_max - ctrl_min) / 100.0  # adjust steps as needed

# Create lists to store the control values and resulting joint angles
control_values = []
joint_angles = []  # will store joint angle in degrees

# Set the simulation to run without delays to speed up the test
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Sweep control signal from min to max
    control = ctrl_min
    increasing = True
    
    # Run for a fixed number of simulation steps
    for step in range(1000):
        # Set control for actuator 0, and zero for others
        data.ctrl[0] = control
        for i in range(1, num_actuators):
            data.ctrl[i] = 0.0
        
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Record the control value and the corresponding joint position
        control_values.append(control)
        joint_angle_deg = np.degrees(data.qpos[0])  # joint angle in degrees for actuator 0
        joint_angles.append(joint_angle_deg)
        
        # Update control value: sweep between ctrl_min and ctrl_max
        if increasing:
            control += step_size
            if control >= ctrl_max:
                control = ctrl_max
                increasing = False
        else:
            control -= step_size
            if control <= ctrl_min:
                control = ctrl_min
                increasing = True
        
        t += 0.005  # Increase time (no sleep for fast simulation)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(control_values, joint_angles, 'o-', label='Joint 0 angle (deg)')
plt.xlabel('Control Signal (command)')
plt.ylabel('Joint Angle (degrees)')
plt.title('Empirical Mapping from Control Signal to Joint Angle')
plt.legend()
plt.grid(True)
plt.show()

# Print observed min and max angles
print("Observed Joint Angle Limits for Actuator 0 (in degrees):")
print(f"Minimum: {min(joint_angles):.2f}, Maximum: {max(joint_angles):.2f}")
