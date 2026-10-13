from scipy.optimize import root
import numpy
import keyboard
k= 0.5
def equations(vars):
    x,y=vars
    eq1= y-2*numpy.sin(k*x)
    eq2 = ((x-x0)**2+(y-y0)**2)**0.5-l
    return [eq1,eq2]

def points(x0,y0):
    sol = root(equations,[x0,y0])
    return tuple(sol.x)
x0 = 0
y0 = 0

l=1
sol = points(x0,y0)
x0=sol[0]
y0=sol[1]
print(sol)
num_rods=1
x_temp = 0
y_temp = 0
max_rods = 10
print("hi")
while(not(abs(x0-2*numpy.pi/k)<0.15 and abs(y0)<0.2)):
    sol = points(x0,y0)
    x0=sol[0]
    y0=sol[1]
    if(x0>2*numpy.pi/k+0.1):
        print(num_rods,"rods of length",l,"and",k*x0/(numpy.pi),"pi lengths,","k is ",k)
        print("failed, need to extend sine wave")
        k=k-0.05
        x0=0
        y0=0
        x_temp,y_temp = points(x0,y0)
        while(x_temp<x0):
            x_temp, y_temp = points(x0,y0)
        x0,y0=x_temp,y_temp
        num_rods=1
    print(sol)
    num_rods = num_rods+1
    if(num_rods>max_rods):
        print(num_rods,"rods of length",l,"and",k*x0/(numpy.pi),"pi lengths")
        print("failed, need to shorten wave")
        k=k+0.025
        x0=0
        y0=0
        x_temp,y_temp = points(x0,y0)
        while(x_temp<x0):
            x_temp, y_temp = points(x0,y0)
        x0,y0=x_temp,y_temp
        num_rods=1
    # if(num_rods>50):
    #     print("figure out why i didnt work")
    #     break
    if(keyboard.is_pressed('q')):
        break
print(num_rods,"rods of length",l,"and",k*x0/(numpy.pi),"pi lengths")

    