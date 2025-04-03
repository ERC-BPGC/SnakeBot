'''
new estimation method, sacrifices some accuracy for simpler and faster code
'''
import numpy as np
import matplotlib.pyplot as plt
from math import atan
from math import sqrt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev



# Example of how to call the points function
# spline_func should be a function that returns x, y, z given a parameter `u`
def spline_func(u):
    return splev(u, tck)  # Assuming you have tck from splprep

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))
# Initial point on the curve
x0, y0, z0 = 0, 0, 0  # You should choose an appropriate starting point

n=1
answers = []
error_margin=5
still_not_found =True
answers.append([x0,y0,z0])
colors =['red','blue']

# User inputs for required number of segments, length, amplitudes, wavelength, and error margin
# req_rods=int(input("Enter the required number of segments: "))
req_length = float(input("Enter the required length between two servo joints: "))
req_amplitude_z = float(input("Enter the required amplitude in the z axis:"))
req_amplitude_y = float(input("Enter the required amplitude in the y axis:"))
want_sine_wave = input("do u want the sine wave plotted as well?(y/n)")


num_path_points = int(input("Enter the number of points: "))

# Initialize lists to store x and y coordinates
path_points_x = [0]
path_points_y = [0]

# Take input for the coordinates
for i in range(num_path_points):
    x = float(input(f"Enter x-coordinate of point {i+1}: "))
    y = float(input(f"Enter y-coordinate of point {i+1}: "))
    path_points_x.append(x)
    path_points_y.append(y)


# Convert lists to a numpy array
points = np.array([path_points_x, path_points_y])

# Create a parameterized spline
tck, u = splprep(points, s=0)

# Generate points along the spline
u_new = np.linspace(0, 1, 100000)  # 10000 points along the spline, to increase accuracy(making it slower) just make that number larger(probably alsso change u from 0 to 1, to 2)
x_new, y_new = splev(u_new, tck)

# Calculate the derivatives (tangent vectors) along the spline
dx, dy = splev(u_new, tck, der=1)

# Normalize the tangent vectors
magnitude = np.sqrt(dx**2 + dy**2)
dx_normalized = -dy / magnitude
dy_normalized = dx / magnitude

# Set initial amplitude and frequency
amplitude = 0.5
frequency = float(input("Enter the frequency(number of sine waves ): ")) #I know that ive named it wavelength, but for some reason still sort of acting like frequency

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.25)

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1 in x:y:z

# Plot the original spline in 3D (XY plane)
ax.plot(x_new, y_new, np.zeros_like(x_new), label='Original Spline', color='red')


#plotting sine wave
# Initial sine wave plot in 3D
x_sine_wave = x_new + req_amplitude_y * np.sin(2 * np.pi *frequency * u_new) * dx_normalized
y_sine_wave = y_new + req_amplitude_y * np.sin(2 * np.pi *frequency * u_new) * dy_normalized
z_wave = req_amplitude_z * np.abs(np.sin(2 * np.pi *frequency * u_new))

sine_wave_line, = ax.plot(x_sine_wave, y_sine_wave, z_wave, label='3D Spline with Sine Waves', color='green')
print(type(x_sine_wave),type(y_sine_wave))
print(x_sine_wave[1])

xyz = [[x_sine_wave[i], y_sine_wave[i], z_wave[i]] for i in range(len(x_sine_wave))]
xyz2=np.transpose(np.array([x_sine_wave,y_sine_wave,z_wave]))
answers.append(np.array([0,0,0]))
print(xyz2[-1])
for i in range(len(x_sine_wave)):
    if(sqrt((x_sine_wave[i]-answers[-1][0])**2+(y_sine_wave[i]-answers[-1][1])**2+(z_wave[i]-answers[-1][2])**2)>req_length):
        answers.append([x_sine_wave[i],y_sine_wave[i],z_wave[i]])
    # else:
        # print("hi")
answers=np.array(answers)
# print(answers)
ans_x=answers[:,0]
ans_y=answers[:,1]
ans_z=answers[:,2]
# print(ans_x)
ax.plot(ans_x,ans_y,ans_z,'bo-',label=f'Segments along path')
max_dist =max(max(path_points_x),max(path_points_y))
ax.set_xlim(0,max_dist)
ax.set_ylim(0,max_dist)
ax.set_zlim(0,max_dist)

angles_ground_apparent =[]
angles_ground_real=[]
angles_relative=[]
angles_real =[]
for i in range(0,len(answers)-1):
    # Calculate angles relative to the ground and relative to previous segment
    angles_ground_apparent.append([atan((answers[i+1][2]-answers[i][2])/(answers[i+1][0]-answers[i][0]))*180/np.pi,atan((answers[i+1][1]-answers[i][1])/(answers[i+1][0]-answers[i][0]))*180/np.pi])
    #still needs to be verified if this is correct way of finding the real angles
    angles_ground_real.append([atan((answers[i+1][2]-answers[i][2])/sqrt((answers[i+1][0]-answers[i][0])**2+(answers[i+1][1]-answers[i][1])**2))*180/np.pi,atan((answers[i+1][1]-answers[i][1])/sqrt((answers[i+1][0]-answers[i][0])**2+(answers[i+1][2]-answers[i][2])**2))*180/np.pi])
    if i:
        angles_relative.append(tuple([round(180+angles_ground_apparent[i][0]-angles_ground_apparent[i-1][0],3),round(180+angles_ground_apparent[i][1]-angles_ground_apparent[i-1][1],3)]))
        angles_real.append(tuple([round(180+angles_ground_real[i][0]-angles_ground_real[i-1][0],3),round(180+angles_ground_real[i][1]-angles_ground_real[i-1][1],3)]))
print("The real angles are:")
for item in angles_real:
    print(item[0])
    print(item[1])
# print("The apparent angles are:")
# for item in angles_relative:
    # print(item)

