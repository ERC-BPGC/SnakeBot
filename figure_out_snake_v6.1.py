from scipy.optimize import fsolve
import numpy as np
from math import atan
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import differential_evolution

# Define the system of equations to solve
def equations(vars, x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length):
    x, y, z = vars
    eq1 = y - req_amplitude_y * np.sin(2 * np.pi * x / wavelength)
    eq2 = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) - req_length
    eq3 = z - np.abs(req_amplitude_z * np.sin(2 * np.pi * x / wavelength))
    return eq1**2 + eq2**2 + eq3**2

# Validate if the found solution matches the expected sine wave
def validate_solution(x, y, z, req_amplitude_y, req_amplitude_z, wavelength):
    y_expected = req_amplitude_y * np.sin(2 * np.pi * x / wavelength)
    z_expected = np.abs(req_amplitude_z * np.sin(2 * np.pi * x / wavelength))
    return np.isclose(y, y_expected, atol=1e-3) and np.isclose(z, z_expected, atol=1e-3)

# Function to find the next point (x, y, z) on the sine wave
def points(x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length):
    # Define the search bounds for the differential evolution algorithm
    bounds = [(x0 + 0.01, x0 + 3), (y0 - 3, y0 + 3), (z0 - 3, z0 + 3)]
    # Use differential evolution to minimize the system of equations
    result = differential_evolution(equations, bounds, args=(x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length), tol=1e-8)
    x, y, z = result.x
    # Validate the solution and return it, or recursively call the function again if its invalid
    if validate_solution(x, y, z, req_amplitude_y, req_amplitude_z, wavelength):
        return tuple(result.x)
    else:
        return points(x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length)

#Starting point(origin)
x0 = 0
y0 = 0
z0 = 0

answers = []
error_margin=5
still_not_found =True
answers.append([x0,y0,z0])


colors =['green','red','blue']

# User inputs for required number of segments, length, amplitudes, wavelength, and error margin
'''
NOTE
The code may not function properly if the segment length and amplitudes are too large.
For example, if the segment length is set to 6, the amplitudes set to 8, andwavelength 25, the calculations may not converge or yield the desired results.
It is recommended to scale down the inputs proportionally. For instance, instead of using 6, 8, 8, you can use 0.6, 0.8, 0.8,2.5 respectively.
'''
req_rods=int(input("Enter the required number of segments: "))
req_length = float(input("Enter the required length between two servo joints: "))
req_amplitude_z = float(input("Enter the required amplitude in the z axis:"))
req_amplitude_y = float(input("Enter the required amplitude in the y axis:"))
wavelength =float(input("Enter the wavelength:"))
error_margin = float(input("set the required percentage error margin(error margin may be increased if a repeating set of angles isnt found to be less than 100+number of segments):"))
want_sine_wave = input("do u want the sine wave plotted as well?(y/n)")


# Loop until a valid solution is found
while(still_not_found):
    for i in range(0,req_rods):
        sol = points(x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length)
        x0=sol[0]
        y0=sol[1]
        z0=sol[2]
        answers.append([x0,y0,z0])

    # Check if the last point meets the error margin criteria
    while(True):
        if(y0/req_amplitude_y<=error_margin/100 and z0/req_amplitude_z <=error_margin/100):
            print("SUCCESS")
            still_not_found=False
            break
        # If too many points are generated without success, increase the error margin and try again
        if(len(answers)>100+req_rods):
            print("Im not sure why this is failing but lemme try increasing the error margin")
            error_margin +=1
            answers.clear()
            x0=0
            y0=0
            z0=0
            answers.append([x0,y0,z0])
            break
        sol = points(x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length)
        x0=sol[0]
        y0=sol[1]
        z0=sol[2]
        answers.append([x0,y0,z0])
        
# Output the final points and results
print("Points on 3d wala graph:")
for item in answers:
    print(tuple(item))
print(req_length)


print("The snake segments will repeat a set of ",len(answers)-2,"angles,",req_rods-1,"angles at a time\nThe starting points are:")
for i in range(0,req_rods):
    print(tuple(answers[i]))
x_plot=[]
y_plot=[]
z_plot=[]
angles_ground_apparent=[]
angles_ground_real =[]
angles_relative =[]
angles_real =[]
# Plot the points and calculate the angles
fig = plt.figure()
ax = plt.axes(projection ='3d')
print("The angles are:")
for i in range(0,len(answers)-1):
    x_test = np.linspace(answers[i][0],answers[i+1][0],100)
    y_test = np.linspace(answers[i][1],answers[i+1][1],100)
    z_test = np.linspace(answers[i][2],answers[i+1][2],100)

    # Calculate angles relative to the ground and relative to previous segment
    angles_ground_apparent.append([atan((answers[i+1][2]-answers[i][2])/(answers[i+1][0]-answers[i][0]))*180/np.pi,atan((answers[i+1][1]-answers[i][1])/(answers[i+1][0]-answers[i][0]))*180/np.pi])
    #still needs to be verified if this is correct way of finding the real angles
    angles_ground_real.append([atan((answers[i+1][2]-answers[i][2])/sqrt((answers[i+1][0]-answers[i][0])**2+(answers[i+1][1]-answers[i][1])**2)),atan((answers[i+1][1]-answers[i][1])/sqrt((answers[i+1][0]-answers[i][0])**2+(answers[i+1][2]-answers[i][2])**2))])
    if i:
        angles_relative.append(tuple([round(180+angles_ground_apparent[i][0]-angles_ground_apparent[i-1][0],3),round(180+angles_ground_apparent[i][1]-angles_ground_apparent[i-1][1],3)]))
        angles_real.append(tuple([round(180+angles_ground_real[i][0]-angles_ground_real[i-1][0],3),round(180+angles_ground_real[i][1]-angles_ground_real[i-1][1],3)]))

    ax.plot3D(x_test,y_test,z_test,'purple' if i<req_rods else colors[i%3])

# Print the calculated relative angles
print("The relative angles are:")
for item in angles_relative:
    print(item)
print("The real angles are:")

for item in angles_real:
    print(item)
print("The snake segments will repeat a set of ",len(answers)-2,"angles,",req_rods-1,"angles at a time.")

#plotting sine wave
if(want_sine_wave=='y'):
    x_wave = np.linspace(0, answers[-1][0], 1000)
    y_wave = req_amplitude_y * np.sin(2 * np.pi * x_wave / wavelength)
    z_wave = np.abs(req_amplitude_z * np.sin(2 * np.pi * x_wave / wavelength))
    ax.plot3D(x_wave, y_wave, z_wave, 'cyan', label="Sine Wave")



# Set limits for the 3D plot
ax.set_xlim(0,answers[-1][0])
ax.set_ylim(0,answers[-1][0])
ax.set_zlim(0,answers[-1][0])

# Display the plot
plt.show()
