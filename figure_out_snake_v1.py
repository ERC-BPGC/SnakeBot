from scipy.optimize import root
import numpy
from math import atan
k= 2
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
answers.append([x0,y0])
l=1
sol = points(x0,y0)
x0=sol[0]
y0=sol[1]
print(sol)
num_rods=0
req_rods=5
req_length = 2
prev_angle = 180
curr_angle = 0

while(not(abs(x0-2*numpy.pi/k)<0.15 and abs(y0)<0.1)):
    sol = points(x0,y0)
    x0=sol[0]
    y0=sol[1]
    if(x0>2*numpy.pi/k+0.1) or (num_rods>5):
        print("failed")
        answers = []
        l= l+0.1
        x0=0
        y0=0
        answers.append([x0,y0])
        x0,y0=points(x0,y0)
        num_rods=0
    print([x0,y0])
    answers.append([x0,y0])
    num_rods = num_rods+1
    if(num_rods>50):
        print("figure out why i didnt work")
        break
 
print(num_rods,"rods of length",l,"and",k*x0/(numpy.pi),"pi lengths")
print("so the sine wave for rod length",req_length,"must have wavelength =",2*numpy.pi*req_length/l,"and amplitude ",req_length/l,"\n the points are:" )
for item in answers:
    print([item[0]*req_length/l,item[1]*req_length/l])
print("the angles are:")
for i in range(1,len(answers)):
    curr_angle = atan((answers[i][1]-answers[i-1][1])/(answers[i][0])-answers[i-1][0])*180/numpy.pi
    print(curr_angle+180-prev_angle)
    prev_angle = curr_angle 