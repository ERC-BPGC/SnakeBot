import numpy as np
import matplotlib.pyplot as plt
from math import atan, sqrt
from mpl_toolkits import mplot3d
from scipy.interpolate import splprep, splev

# Hardcoded input values
req_length = 10  # Required length between two servo joints
req_amplitude_z = 5  # Required amplitude in the z-axis
req_amplitude_y = 3  # Required amplitude in the y-axis
want_sine_wave = 'y'  # Whether to plot the sine wave
num_path_points = 4  # Number of path points

# Hardcoded path coordinates
path_points_x = [0, 10, 20, 30]
path_points_y = [0, 5, 10, 15]

# Hardcoded sine wave frequency
frequency = 2  # Number of sine wave cycles

# Convert lists to a numpy array
points = np.array([path_points_x, path_points_y])

# Create a parameterized spline
tck, u = splprep(points, s=0)

# Generate points along the spline
u_new = np.linspace(0, 1, 100000)  # 100,000 points along the spline
x_new, y_new = splev(u_new, tck)

# Calculate the derivatives (tangent vectors) along the spline
dx, dy = splev(u_new, tck, der=1)

# Normalize the tangent vectors
magnitude = np.sqrt(dx**2 + dy**2)
dx_normalized = -dy / magnitude
dy_normalized = dx / magnitude

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.25)

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1 in x:y:z

# Plot the original spline in 3D (XY plane)
ax.plot(x_new, y_new, np.zeros_like(x_new), label='Original Spline', color='red')

# Plotting sine wave
x_sine_wave = x_new + req_amplitude_y * np.sin(2 * np.pi * frequency * u_new) * dx_normalized
y_sine_wave = y_new + req_amplitude_y * np.sin(2 * np.pi * frequency * u_new) * dy_normalized
z_wave = req_amplitude_z * np.abs(np.sin(2 * np.pi * frequency * u_new))

sine_wave_line, = ax.plot(x_sine_wave, y_sine_wave, z_wave, label='3D Spline with Sine Waves', color='green')

# Segments based on required length
answers = [[0, 0, 0]]
for i in range(len(x_sine_wave)):
    if sqrt((x_sine_wave[i] - answers[-1][0])**2 +
            (y_sine_wave[i] - answers[-1][1])**2 +
            (z_wave[i] - answers[-1][2])**2) > req_length:
        answers.append([x_sine_wave[i], y_sine_wave[i], z_wave[i]])

answers = np.array(answers)
ans_x = answers[:, 0]
ans_y = answers[:, 1]
ans_z = answers[:, 2]
ax.plot(ans_x, ans_y, ans_z, 'bo-', label=f'Segments along path')

# Set limits for 3D plot
max_dist = max(max(path_points_x), max(path_points_y))
ax.set_xlim(0, max_dist)
ax.set_ylim(0, max_dist)
ax.set_zlim(0, max_dist)

# Calculate angles
angles_ground_apparent = []
angles_ground_real = []
angles_relative = []
angles_real = []
for i in range(0, len(answers) - 1):
    angles_ground_apparent.append([
        atan((answers[i + 1][2] - answers[i][2]) / (answers[i + 1][0] - answers[i][0])) * 180 / np.pi,
        atan((answers[i + 1][1] - answers[i][1]) / (answers[i + 1][0] - answers[i][0])) * 180 / np.pi
    ])
    angles_ground_real.append([
        atan((answers[i + 1][2] - answers[i][2]) / sqrt((answers[i + 1][0] - answers[i][0])**2 + (answers[i + 1][1] - answers[i][1])**2)) * 180 / np.pi,
        atan((answers[i + 1][1] - answers[i][1]) / sqrt((answers[i + 1][0] - answers[i][0])**2 + (answers[i + 1][2] - answers[i][2])**2)) * 180 / np.pi
    ])
    if i:
        angles_relative.append(tuple([
            round(180 + angles_ground_apparent[i][0] - angles_ground_apparent[i - 1][0], 3),
            round(180 + angles_ground_apparent[i][1] - angles_ground_apparent[i - 1][1], 3)
        ]))
        angles_real.append(tuple([
            round(180 + angles_ground_real[i][0] - angles_ground_real[i - 1][0], 3),
            round(180 + angles_ground_real[i][1] - angles_ground_real[i - 1][1], 3)
        ]))

print("The real angles are:")
for item in angles_real:
    print(item)
print("The apparent angles are:")
for item in angles_relative:
    print(item)

# Finalize plot
plt.legend()
plt.grid(True)
plt.show()
