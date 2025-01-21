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
# def spline_func(u):
#     return splev(u, tck)  # Assuming you have tck from splprep

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
req_segments=6#int(input("Enter the required number of segments: "))
req_length = 1#(doesnt matter)#float(input("Enter the required length between two servo joints: "))
req_amplitude_z = 0.2#float(input("Enter the required amplitude in the z axis:"))
req_amplitude_y = 1#float(input("Enter the required amplitude in the y axis:"))
want_sine_wave = 'y'#input("do u want the sine wave plotted as well?(y/n)")

delta=req_length/2
# Set initial amplitude and frequency
wavelength = 4#float(input("Enter the wavelength:"))
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.25)

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1 in x:y:z

# Plot the original spline in 3D (XY plane)
# ax.plot(x_new, y_new, np.zeros_like(x_new), label='Original Spline', color='red')


#plotting sine wave
# Initial sine wave plot in 3D
x_wave=np.linspace(0,2*wavelength,100000)
y_wave = req_amplitude_y * np.sin(2 * np.pi * x_wave / wavelength)
z_wave = np.abs(req_amplitude_z * np.sin(2 * np.pi * x_wave / wavelength))
ax.plot3D(x_wave, y_wave, z_wave, 'cyan', label="Sine Wave")


once = True
temp = 0
num_fails_short=0
num_fails_long=0
l=req_length
while True:
    for i in range(len(x_wave)):
        if((sqrt((x_wave[i]-answers[-1][0])**2+(y_wave[i]-answers[-1][1])**2+(z_wave[i]-answers[-1][2])**2))>l):
            answers.append([x_wave[i],y_wave[i],z_wave[i]])
            temp+=1
            if (temp>=req_segments):
                break
    if(answers[-1][0]>wavelength*1.05):
        # print("failed, too long")
        num_fails_long+=1
        num_fails_short=0
        if num_fails_long>50:
            break
        answers = []
        l=l-delta
        delta=delta/2
        once=False
        x0=0
        y0=0
        z0=0
        temp=0
        answers.append([x0,y0,z0])
        # x0,y0=points(x0,y0)
    elif(answers[-1][0]<wavelength*0.95):
        # print("failed, too short")
        num_fails_short+=1
        num_fails_long=0
        if(num_fails_short>50):
            break
        answers = []
        l=l+delta
        x0=0
        y0=0
        z0=0
        temp=0
        answers.append([x0,y0,z0])
        if(not once):
            delta=delta/2
    else:#its in the acceptable range
        break        

answers=np.array(answers)
# print(answers)
ans_x=answers[:,0]
ans_y=answers[:,1]
ans_z=answers[:,2]
# print(ans_x)
ax.plot(ans_x,ans_y,ans_z,'bo-',label=f'Segments along path')
max_dist =max(wavelength,req_amplitude_y,req_amplitude_z)
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
        angles_real .append(tuple([round(180+angles_ground_real[i][0]-angles_ground_real[i-1][0],3),round(180+angles_ground_real[i][1]-angles_ground_real[i-1][1],3)]))
print("The real angles are:")
for item in angles_real:
    print(item[1])#horizontal
    print(item[0])#vertical
print("The apparent angles are:")
for item in angles_relative:
    print(item[1])#horizontal
    print(item[0])#vertical
# Call update function on slider value change
plt.legend()
plt.grid(True)
plt.show()
