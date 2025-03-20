'''
new estimation method, sacrifices some accuracy for simpler and faster code
NOTE from here on the angles printed are now in (x,y) and not in (y,x) as in the codes before
'''
import numpy as np
import serial
import matplotlib.pyplot as plt
import time 
import math
from math import atan,sin, cos, sqrt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev
from copy import deepcopy
import keyboard
# spline_func should be a function that returns x, y, z given a parameter `u`
def spline_func(u):
    return splev(u, tck)  # Assuming you have tck from splprep

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))
# Initial point on the curve
x0, y0, z0 = 0, 0, 0  # You should choose an appropriate starting point

n=1
iterations =0
answers = []
error_margin=5
accuracylevel =100000
still_not_found =True
colors =['red','blue']

# User inputs for required number of segments, length, amplitudes, wavelength, and error margin
num_segments=5#int(input("Enter the required number of segments: "))+1
req_length = 1.14#cad measured value #float(input("Enter the required length between two servo joints: "))
req_amplitude_z = 0.8#float(input("Enter the required amplitude in the z axis:"))
req_amplitude_y = 1.1#float(input("Enter the required amplitude in the y axis:"))
A1=0
num_path_points = 0
R=req_length/2#radius of the virtual rolling joint approximate sphere
approximation_circle_dist=0.17
# Initialize lists to store x and y coordinates, starts from origin
path_points_x = [0,3,4,6]
path_points_y = [0,3,4,9]
frequency = 2#float(input("Enter the frequency(number of sine waves ): ")) #I know that ive named it wavelength, but for some reason still sort of acting like frequency

# Take input for the coordinates
# for i in range(num_path_points):
#     x_coord = float(input(f"Enter x-coordinate of point {i+1}: "))
#     y_coord = float(input(f"Enter y-coordinate of point {i+1}: "))
#     path_points_x.append(x_coord)
#     path_points_y.append(y_coord)


# Convert lists to a numpy array
points = np.array([path_points_x, path_points_y])

# Create a parameterized spline
tck, u = splprep(points, s=0)

# Generate points along the spline
u_new = np.linspace(0, 1, accuracylevel)  # 10000 points along the spline, to increase accuracy(making it slower) just make that number larger(probably alsso change u from 0 to 1, to 2)
x_new, y_new = splev(u_new, tck)

# Calculate the derivatives (tangent vectors) along the spline
dx, dy = splev(u_new, tck, der=1)

# Normalize the tangent vectors
magnitude = np.sqrt(dx**2 + dy**2)
dx_normalized = -dy / magnitude
dy_normalized = dx / magnitude

# Set initial amplitude and frequency

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
x_sine_wave =A1+ x_new + req_amplitude_y * np.sin(2 * np.pi *frequency * u_new) * dx_normalized
y_sine_wave = A1+y_new + req_amplitude_y * np.sin(2 * np.pi *frequency * u_new) * dy_normalized
z_wave = req_amplitude_z * np.abs(np.sin(2 * np.pi *frequency * u_new))

sine_wave_line, = ax.plot(x_sine_wave, y_sine_wave, z_wave, label='3D Spline with Sine Waves', color='green')
print(type(x_sine_wave),type(y_sine_wave))
print(x_sine_wave[1])


xyz = [[x_sine_wave[i], y_sine_wave[i], z_wave[i]] for i in range(len(x_sine_wave))]
xyz2=np.transpose(np.array([x_sine_wave,y_sine_wave,z_wave]))
print(xyz2[-1])


angles_ground_apparent =[]
angles_ground_real=[]
angles_relative=[]
angles_real =[]
answers_per_iteration =[]
angles_relative_per_iteration=[]
angles_real_per_iteration=[]

offset =0
answers_per_iteration.append([0,0,0])
while(offset<accuracylevel):
    # print("hi", end=':')
    for i in range(offset,len(x_sine_wave)):
        if(sqrt((x_sine_wave[i]-answers_per_iteration[-1][0])**2+(y_sine_wave[i]-answers_per_iteration[-1][1])**2+(z_wave[i]-answers_per_iteration[-1][2])**2)>req_length):
            answers_per_iteration.append([x_sine_wave[i],y_sine_wave[i],z_wave[i]])
        if(len(answers_per_iteration)==num_segments+1):
            # Calculate angles relative to the ground and relative to previous segment
            # answers_per_iteration=np.array(answers_per_iteration)
            for j in range(0,len(answers_per_iteration)-1):
                angles_ground_apparent.append([atan((answers_per_iteration[j+1][1]-answers_per_iteration[j][1])/(answers_per_iteration[j+1][0]-answers_per_iteration[j][0]))*180/np.pi,atan((answers_per_iteration[j+1][2]-answers_per_iteration[j][2])/(answers_per_iteration[j+1][0]-answers_per_iteration[j][0]))*180/np.pi])
                #still needs to be verified if this is correct way of finding the real angles
                angles_ground_real.append([atan((answers_per_iteration[j+1][1]-answers_per_iteration[j][1])/sqrt((answers_per_iteration[j+1][0]-answers_per_iteration[j][0])**2+(answers_per_iteration[j+1][2]-answers_per_iteration[j][2])**2))*180/np.pi,atan((answers_per_iteration[j+1][2]-answers_per_iteration[j][2])/sqrt((answers_per_iteration[j+1][0]-answers_per_iteration[j][0])**2+(answers_per_iteration[j+1][1]-answers_per_iteration[j][1])**2))*180/np.pi])
                if j:
                    angles_relative_per_iteration.append(tuple([round(180+angles_ground_apparent[j][0]-angles_ground_apparent[j-1][0],3),round(180+angles_ground_apparent[j][1]-angles_ground_apparent[j-1][1],3)]))
                    angles_real_per_iteration.append(tuple([round(180+angles_ground_real[j][0]-angles_ground_real[j-1][0],3),round(180+angles_ground_real[j][1]-angles_ground_real[j-1][1],3)]))
            break
    # print("This: ", angles_real_per_iteration)
    # print(angles_real_per_iteration)
    # sleep(0.065)
    if(len(answers_per_iteration)<num_segments+1):
        break
    # print("Then This appends...")
    answers.append(deepcopy(answers_per_iteration))#normal copy was giving some issues


    angles_real.append(deepcopy(angles_real_per_iteration))
    # print(angles_real)
    angles_relative.append(deepcopy(angles_relative_per_iteration))
    answers_per_iteration.clear()
    angles_real_per_iteration.clear()
    angles_ground_apparent.clear()
    angles_ground_real.clear()


    angles_relative_per_iteration.clear()
    offset+=int(accuracylevel/150)
    answers_per_iteration.append([x_sine_wave[offset],y_sine_wave[offset],z_wave[offset]])

# for item in answers:
#     print(item)

for item in angles_real:
    print(item)#(horizontal,vertial)


print(len(angles_real))



# answers=np.array(answers)
# # print(answers)
# ans_x=answers[:,0]
# ans_y=answers[:,1]
# ans_z=answers[:,2]
# # print(ans_x)
# ax.plot(ans_x,ans_y,ans_z,'bo-',label=f'Segments along path')
max_dist =max(max(path_points_x),max(path_points_y))
ax.set_xlim(0,max_dist)
ax.set_ylim(0,max_dist)
ax.set_zlim(0,max_dist)

# plotting animation wali cheez

line, = ax.plot([], [], [], 'o-', lw=2)
def update(frame):
    """Update function for animation"""
    data = np.array(answers[frame])  # Get points for the current frame
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    # Update line data
    line.set_data(x, y)
    line.set_3d_properties(z)
    return line,

ani = FuncAnimation(fig, update, frames=len(answers), interval=50, blit=False)

plt.legend()
plt.grid(True)
plt.show()

# PySerial for iteratively sending the angles to Arduino board 
ser = serial.Serial('COM8', 230400, timeout=1)
time.sleep(2)  # Wait for connection to stabilize


tempcount = 1
wantToSend=True
while(wantToSend):

    for angles_set in angles_real:  # Iterate over the angle sets
        angle_data= ""
        for angle_pair in angles_set:  # Each set contains tuples of two angles
            '''this modified angle is due to the design of the modules in v2(virtual rolling joint)'''
            angle_1 = atan(R*sin((180-angle_pair[0])*np.pi/180)/(R*cos((180-angle_pair[0])*np.pi/180)-approximation_circle_dist))*180/np.pi
            angle_2 = atan(R*sin((180-angle_pair[1])*np.pi/180)/(R*cos((180-angle_pair[1])*np.pi/180)-approximation_circle_dist))*180/np.pi
            # print(angle_1,"meow",angle_2)
            angle_data += f"{min(110,max(90-int(angle_1),70))}/{min(110,max(90-int(angle_2),70))},"  # Format as comma-separated values ***(horizontal,vertical)***
        angle_data+="\n"
        ser.write(angle_data.encode())  # Send as bytes
        # ser.write("\n".encode())
        print(angle_data, tempcount)
        time.sleep(0.055)
        tempcount+=1
        if(keyboard.is_pressed('p')):
            print("paused")
            time.sleep(2)

            while(True):
                if(keyboard.is_pressed('p')):
                    print("continuing")
                    time.sleep(2)
                    break
                if(keyboard.is_pressed('q')):
                    print("quitting")
                    wantToSend=False
                    break
        if(not wantToSend):
            break
        # response = ser.readline().decode().strip()  # Read   serial buffer5
        # time.sleep(10)
        # if response:    
        #     print(f"Received: {response}")  # Print received data
    # print("Serial communication finished.")
    # if(input("Press R to restart sending angle commands: ").upper()!='R'):
    #     print("Ending Execution")
    #     ser.close()  # Close serial connection
    #     break
    # else :
    #     print("Restarting angle commands")
