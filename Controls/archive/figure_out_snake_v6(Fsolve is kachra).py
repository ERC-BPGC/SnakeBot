from scipy.optimize import fsolve
import numpy as np
from math import atan
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
from scipy.optimize import differential_evolution


colors =['green','red','blue']
delta_guess=0.1# have to vary using A and K
req_amplitude_z =1
req_amplitude_y =1
wavelength =1
answers = []
error_margin =10#percentage error
count = 0
def equations(vars, x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length):
    x, y, z = vars
    eq1 = y - req_amplitude_y * np.sin(2 * np.pi * x / wavelength)
    eq2 = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) - req_length
    eq3 = z - np.abs(req_amplitude_z * np.sin(2 * np.pi * x / wavelength))
    return eq1**2 + eq2**2 + eq3**2

def validate_solution(x, y, z, req_amplitude_y, req_amplitude_z, wavelength):
    y_expected = req_amplitude_y * np.sin(2 * np.pi * x / wavelength)
    z_expected = np.abs(req_amplitude_z * np.sin(2 * np.pi * x / wavelength))
    return np.isclose(y, y_expected, atol=1e-3) and np.isclose(z, z_expected, atol=1e-3)

def points(x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length):
    bounds = [(x0 + 0.01, x0 + 3), (y0 - 3, y0 + 3), (z0 - 3, z0 + 3)]
    result = differential_evolution(equations, bounds, args=(x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length), tol=1e-8)
    x, y, z = result.x
    if validate_solution(x, y, z, req_amplitude_y, req_amplitude_z, wavelength):
        return tuple(result.x)
    else:
        return points(x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length)
x0 = 0
y0 = 0
z0 = 0
answers.append([x0,y0,z0])

req_rods=int(input("Enter the required number of segments: "))
req_length = float(input("Enter the required length between two servo joints: "))
req_amplitude_z = float(input("Enter the required amplitude in the z axis:"))
req_amplitude_y = float(input("Enter the required amplitude in the y axis:"))
wavelength =float(input("Enter the wavelength:"))
# error_margin = int(input("set the required percentage error margin:"))
# delta_guess=0.5*wavelength*wavelength/(req_amplitude_y*req_amplitude_z)
want_sine_wave = input("do u want the sine wave plotted as well?(y/n)")
for i in range(0,req_rods):
    sol = points(x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length)
    x0=sol[0]
    y0=sol[1]
    z0=sol[2]
    answers.append([x0,y0,z0])
# while(True):
#     if(y0/req_amplitude_y<=error_margin/100 and z0/req_amplitude_z <=error_margin/100):
#         print("SUCCESS")

#         break
#     if(len(answers)>200):
#         print("Im not sure why this is failing but ")
#         break
#     sol = points(x0, y0, z0, req_amplitude_y, req_amplitude_z, wavelength, req_length)
#     x0=sol[0]
#     y0=sol[1]

#     z0=sol[2]
#     answers.append([x0,y0,z0])
#     print(tuple([x0,y0,z0]))
    # print("trying\n")
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
angles_ground=[]
angles_relative =[]

fig = plt.figure()
ax = plt.axes(projection ='3d')
print("The angles are:")
for i in range(0,len(answers)-1):
    x_test = np.linspace(answers[i][0],answers[i+1][0],100)
    y_test = np.linspace(answers[i][1],answers[i+1][1],100)
    z_test = np.linspace(answers[i][2],answers[i+1][2],100)
    angles_ground.append([atan((answers[i+1][2]-answers[i][2])/(answers[i+1][0]-answers[i][0]))*180/np.pi,atan((answers[i+1][1]-answers[i][1])/(answers[i+1][0]-answers[i][0]))*180/np.pi])
    if i:
        angles_relative.append(tuple([round(180+angles_ground[i][0]-angles_ground[i-1][0],3),round(180+angles_ground[i][1]-angles_ground[i-1][1],3)]))
    ax.plot3D(x_test,y_test,z_test,'purple' if i<req_rods else colors[i%3])
for item in angles_relative:
    print(item)


#plotting sine wave
if(want_sine_wave=='y'):
    x_wave = np.linspace(0, answers[-1][0], 1000)
    y_wave = req_amplitude_y * np.sin(2 * np.pi * x_wave / wavelength)
    z_wave = np.abs(req_amplitude_z * np.sin(2 * np.pi * x_wave / wavelength))
    ax.plot3D(x_wave, y_wave, z_wave, 'cyan', label="Sine Wave")




ax.set_xlim(0,answers[-1][0])
ax.set_ylim(0,answers[-1][0])
ax.set_zlim(0,answers[-1][0])

plt.show()







 



# angles=[]
# angles2=[]

# for i in range(len(answers)-1):
#     angles.append(180*atan((answers[i+1][1]-answers[i][1])/(answers[i+1][0]-answers[i][0]))/math.pi)
#     if i:
#         angles2.append(round(180+angles[i]-angles[i-1],3))

# print(angles2)

# fig,ax=plt.subplots()
# ax.plot(x_plot, y_plot, 'ro-')

# for i in range(len(angles2)):
#     ax.text(answers[i+1][0],answers[i+1][1],angles2[i])
# plt.show()