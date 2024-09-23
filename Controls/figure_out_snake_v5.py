from scipy.optimize import fsolve
import numpy as np
from math import atan
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d


colors =['green','red','blue','yellow']
k=1#do not change
A=1
delta_guess=0.1# have to vary based  on A and K
vertical_scale =1

answers = []

def equations(vars):
    x,y,z=vars
    eq1= y-A*np.sin(k*x)
    eq2 = ((x-x0)**2+(y-y0)**2+(z-z0)**2)**0.5-l
    eq3 =z-np.absolute(A*np.sin(k*x)/vertical_scale)
    return [eq1,eq2,eq3]

def points(x0,y0,z0):
    sol = fsolve(equations,[x0+delta_guess,y0,z0])
    return tuple(sol)
x0 = 0
y0 = 0
z0 = 0
once =True
answers.append([x0,y0,z0])
l=1
delta=l/2
num_rods=0
req_rods=int(input("Enter the required number of segments: "))
req_length = float(input("Enter the required length between two servo joints: "))
num_waves = int(input("number of waves of snake:"))
# sine_ratio= float(input("Enter the desired ratio of amplitude to frequency of the sine wave(enter 1 for the generic):"))
while(True):
    for i in range(0,req_rods):
        sol = points(x0,y0,z0)
        x0=sol[0]
        y0=sol[1]
        z0=sol[2]
        answers.append([x0,y0,z0])
    if(x0>num_waves*2*np.pi/k+0.05):
        print("failed, too long")
        answers = []
        l=l-delta
        delta=delta/2
        once=False
        x0=0
        y0=0
        z0=0
        answers.append([x0,y0,z0])
        # x0,y0=points(x0,y0)
    elif(x0<num_waves*2*np.pi/k-0.05):
        print("failed, too short")
        answers = []
        l=l+delta
        x0=0
        y0=0
        z0=0
        answers.append([x0,y0,z0])
        if(not once):
            delta=delta/2
    else:#its in the acceptable range
        break
print("points on 3d wala graph:")
for item in answers:
    print(tuple(item))
print(l)
print(req_rods,"rods of length",l,"and",k*x0/(np.pi),"pi lengths and delta is",delta)
print("so the 3d snake wave for rod length",req_length,"must have wavelength =",2*req_length/l,"pi lengths, and amplitude ",req_length/l,"\nThe points on the modified sine wave are:" )

mod_ans=[]
x_plot=[]
y_plot=[]
z_plot=[]
angles_ground=[]
angles_relative =[]
for item in answers:
    a=[item[0]*req_length/l,item[1]*req_length/l,item[2]*req_length/l]
    x_plot.append(item[0]*req_length/l)
    y_plot.append(item[1]*req_length/l)
    z_plot.append(item[2]*req_length/l)   
    print(tuple(a))
    mod_ans.append(a)
fig = plt.figure()
ax = plt.axes(projection ='3d')
print("The angles are:")
for i in range(0,req_rods):
    x_test = np.linspace(mod_ans[i][0],mod_ans[i+1][0],100)
    y_test = np.linspace(mod_ans[i][1],mod_ans[i+1][1],100)
    z_test = np.linspace(mod_ans[i][2],mod_ans[i+1][2],100)
    angles_ground.append([atan((mod_ans[i+1][2]-mod_ans[i][2])/(mod_ans[i+1][0]-mod_ans[i][0]))*180/np.pi,atan((mod_ans[i+1][1]-mod_ans[i][1])/(mod_ans[i+1][0]-mod_ans[i][0]))*180/np.pi])
    if i:
        angles_relative.append(tuple([round(180+angles_ground[i][0]-angles_ground[i-1][0],3),round(180+angles_ground[i][1]-angles_ground[i-1][1],3)]))
    ax.plot3D(x_test,y_test,z_test,'red')
for item in angles_relative:
    print(item)
ax.set_xlim(0,mod_ans[-1][0])
ax.set_ylim(0,mod_ans[-1][0])
ax.set_zlim(0,mod_ans[-1][0])
plt.show()







 



# angles=[]
# angles2=[]

# for i in range(len(mod_ans)-1):
#     angles.append(180*atan((mod_ans[i+1][1]-mod_ans[i][1])/(mod_ans[i+1][0]-mod_ans[i][0]))/math.pi)
#     if i:
#         angles2.append(round(180+angles[i]-angles[i-1],3))

# print(angles2)

# fig,ax=plt.subplots()
# ax.plot(x_plot, y_plot, 'ro-')

# for i in range(len(angles2)):
#     ax.text(mod_ans[i+1][0],mod_ans[i+1][1],angles2[i])
# plt.show()
