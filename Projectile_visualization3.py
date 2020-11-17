#!/usr/bin/env python
# coding: utf-8

# - Dmitri LaBelle 
# - Hw 6 Problem 3 
# - The purpose of this code is to replicate 2 dimensional trajectories of projectiles on a planet. Now using an added term r (distance from the surface of the planet) to find time of flight, range, and max height above the surface of the planet. Part c explores the nessicary starting angle the projectile must have to land at the spot (R,0) from it's starting position (0,R). For this I use a nested while loop and terminate the loop when the distance between the two points are minimized. 

# In[2]:


#part a & b
import sys
from math import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Gravity, constant acceleration
g     = 9.8

# Default options:
alpha = 0.0
beta  = 0.0
R = 6.4e6
theta = 45.0				# (degrees)
v0    = 7500.0				# unit: m/s 

GM = g * R**2

def acc(x, y, vx, vy, t, r):

    ax = -(GM * x)/ r**3

    # if (alpha > 0.0): ax += ...
    # if (beta > 0.0): ax += ...
    
    ay = -(GM * y)/ r**3

    # if (alpha > 0.0): ay += ...
    # if (beta > 0.0): ay += ...
    
    return ax,ay


def take_a_step(x, y, vx, vy, t, dt, r):

    ax,ay = acc(x, y, vx, vy, t, r)

    # Predict: 

    x  += vx*dt + 0.5*ax*dt*dt
    y  += vy*dt + 0.5*ay*dt*dt
    vx += ax*dt
    vy += ay*dt
    t  += dt
    r = sqrt(x**2+y**2)

    # Correct: 

    ax1,ay1 = acc(x, y, vx, vy, t, r)

    vx += 0.5*(ax1-ax)*dt
    vy += 0.5*(ay1-ay)*dt
    

    return x,y,vx,vy,t,r


def interp(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)



# Set parameters governing the numerical details 
dt = 1

# Set initial position and velocity 
x0 = 0.0
y0 = R
r0 = sqrt(x0**2+y0**2)

# Determine components of the initial velocity vector 
vx0 = v0 * cos(theta * pi/180.0)
vy0 = v0 * sin(theta * pi/180.0)

# Initialize the trajectory 
t  = 0
x  = x0
y  = y0
r = r0
vx = vx0
vy = vy0
tprev = t
xprev = x
yprev = y
rprev = r

xplot = [x]
yplot = [y]
#print(x, y, t)
rmax = 0.0

while r >= R:

    tprev = t
    xprev = x
    yprev = y
    rprev = r

    (x,y,vx,vy,t,r) = take_a_step(x, y, vx, vy, t, dt, r)
    
    xplot.append(x)
    yplot.append(y)

    #print(x, y, t)
    if r > rmax: 
        rmax = r

tof = interp(rprev, tprev, r, t, R)
phi = atan(interp(rprev, xprev, r, x, R) /interp(rprev, yprev, r, y, R))

the_range = phi * R

print('Initial Velocity =', v0, 'Initial angle =', theta, file=sys.stderr)
print('Time of flight =', tof, 'Range =', the_range, 'Max height=', rmax - R , file=sys.stderr)

plt.plot(xplot,yplot, 'r')
#earth
ax = plt.gca()
circ = mpatches.Circle((0, 0), R, linestyle='solid', edgecolor='b', facecolor='blue')
ax.add_patch(circ)

scale = 1.7
ax.set_xlim(-scale* R ,scale* R)
ax.set_ylim(-scale* R ,scale* R)
ax.set_aspect('equal')
#earth
plt.show()


# In[42]:


#part c

import sys
from math import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Gravity, constant acceleration
g     = 9.8

# Default options:
alpha = 0.0
beta  = 0.0
R = 6.4e6
theta = 36.3				# (degrees)
v0    = 7500.0				# unit: m/s 

GM = g * R**2

def acc(x, y, vx, vy, t, r):

    ax = -(GM * x)/ r**3

    # if (alpha > 0.0): ax += ...
    # if (beta > 0.0): ax += ...
    
    ay = -(GM * y)/ r**3

    # if (alpha > 0.0): ay += ...
    # if (beta > 0.0): ay += ...
    
    return ax,ay


def take_a_step(x, y, vx, vy, t, dt, r):

    ax,ay = acc(x, y, vx, vy, t, r)

    # Predict: 

    x  += vx*dt + 0.5*ax*dt*dt
    y  += vy*dt + 0.5*ay*dt*dt
    vx += ax*dt
    vy += ay*dt
    t  += dt
    r = sqrt(x**2+y**2)

    # Correct: 

    ax1,ay1 = acc(x, y, vx, vy, t, r)

    vx += 0.5*(ax1-ax)*dt
    vy += 0.5*(ay1-ay)*dt
    

    return x,y,vx,vy,t,r


def interp(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)



# Set parameters governing the numerical details 
dt = 1

theta = 30
dtheta = .01
thetamax = 40

dist = 0
distp = 0
distpp = 0
tof = 0
phi = 0
the_range = 0


while theta <= thetamax:
    distpp = distp
    distp = dist
    # Set initial position and velocity 
    x0 = 0.0
    y0 = R
    r0 = sqrt(x0**2+y0**2)

    # Determine components of the initial velocity vector 
    vx0 = v0 * cos(theta * pi/180.0)
    vy0 = v0 * sin(theta * pi/180.0)

    # Initialize the trajectory 
    t  = 0
    x  = x0
    y  = y0
    r = r0
    vx = vx0
    vy = vy0
    tprev = t
    xprev = x
    yprev = y
    rprev = r

    xplot = [x]
    yplot = [y]
    #print(x, y, t)
    rmax = 0.0

    while r >= R:

        tprev = t
        xprev = x
        yprev = y
        rprev = r

        (x,y,vx,vy,t,r) = take_a_step(x, y, vx, vy, t, dt, r)

        xplot.append(x)
        yplot.append(y)

        #print(x, y, t)
        if r > rmax: 
            rmax = r

    dist = sqrt((R-x)**2+(0-y)**2)
    
    if distpp < distp and distp < dist and (distp!=0) and (distpp!=0):
        tof = interp(rprev, tprev, r, t, R)
        phi = atan(interp(rprev, xprev, r, x, R) /interp(rprev, yprev, r, y, R))

        the_range = phi * R
        #print('theta =', theta, file=sys.stderr)
        break
        
    theta = theta + dtheta

print('Initial Velocity =', v0, 'Initial angle =', theta, file=sys.stderr)
print('Time of flight =', tof, 'Range =', the_range, 'Max height=', rmax - R , file=sys.stderr)
print('Distance between landing spot and (R,0) =', distpp,file=sys.stderr )

plt.plot(xplot,yplot, 'r')
#earth
ax = plt.gca()
circ = mpatches.Circle((0, 0), R, linestyle='solid', edgecolor='b', facecolor='blue')
ax.add_patch(circ)

scale = 1.7
ax.set_xlim(-scale* R ,scale* R)
ax.set_ylim(-scale* R ,scale* R)
ax.set_aspect('equal')
#earth
plt.show()


# In[ ]:




