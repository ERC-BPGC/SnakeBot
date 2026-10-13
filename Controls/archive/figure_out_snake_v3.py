from scipy.optimize import root
import numpy
from math import atan
k= 1
answers = []
def equations(vars):
    x,y=vars
    eq1= y-numpy.sin(k*x)
    eq2 = ((x-x0)**2+(y-y0)**2)**0.5-l
    return [eq1,eq2]

def points(x0,y0):
    sol = root(equations,[x0,y0])
    return tuple(sol.x)
x0 = 0
y0 = 0
once =True
answers.append([x0,y0])
l=1
delta=0.5
num_rods=0
req_rods=int(input("enter the required number of segments: "))
req_length = float(input("enter the required length between two servo joints: "))
while(True):
    for i in range(0,req_rods):
        sol = points(x0,y0)
        x0=sol[0]
        y0=sol[1]
        answers.append([x0,y0])
    if(x0>2*numpy.pi/k+0.05):
        print("failed, too long")
        answers = []
        l=l-delta
        delta=delta/2
        once=False
        x0=0
        y0=0
        answers.append([x0,y0])
        # x0,y0=points(x0,y0)
    elif(x0<2*numpy.pi/k-0.05):
        print("failed, too short")
        answers = []
        l=l+delta
        x0=0
        y0=0
        answers.append([x0,y0])
        # x0,y0=points(x0,y0)
        if(not once):
            delta=delta/2
    else:#its in the acceptable range
        break
print("points on simple y=sin(x):")
for item in answers:
    print(item)
print(num_rods,"rods of length",l,"and",k*x0/(numpy.pi),"pi lengths and delta is",delta)
print("so the sine wave for rod length",req_length,"must have wavelength =",2*req_length/l,"pi lengths, and amplitude ",req_length/l,"\n the points on the modified sine wave are:" )
for item in answers:
    print([item[0]*req_length/l,item[1]*req_length/l])