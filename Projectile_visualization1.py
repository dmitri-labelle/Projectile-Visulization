#!/usr/bin/env python
# coding: utf-8

# - Dmitri LaBelle 
# - Hw 6 Problem 1 
# - The purpose of this code is to visualize a projectile moving in two dimensions, effected by indipendant accelerations in the x and y directions. From here we can find the time of flight, max height, and rage covered by thr projectile. We can also then expand the simulation to involve a beta term, which simulates the effects of air resistance. Part e explores the relationship between taking off angle and landing angle. Part c attemps to find the starting velocity nessiacary to restore the range when air resistance was negligable (beta = 0). Part d attemps to find the take of angle which maximizes range, while keeping a single starting velocity. 

# In[2]:


#part a & b
import math
import matplotlib.pyplot as plt


# acceleration due to gravity
g = 9.8
beta = 0 #0.001  #<=== Beta

def accx(x, y, vx, vy, t):
    return  - beta * math.sqrt((vx**2)+(vy**2)) * vx

def accy(x, y, vx, vy, t):
    return  -g - beta * math.sqrt((vx**2)+(vy**2)) * vy

def interp(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def take_a_step(x, y, vx, vy, t, dt):

    ax = accx(x, y, vx, vy, t)
    ay = accy(x, y, vx, vy, t)

    # Predict: 
    x  += vx*dt + 0.5*ax*dt*dt
    y  += vy*dt + 0.5*ay*dt*dt
    vx += ax*dt
    vy += ay*dt

    # Correct: 
    ax1 = accx(x, y, vx, vy, t)
    ay1 = accy(x, y, vx, vy, t)

    vx += 0.5*(ax1-ax)*dt
    vy += 0.5*(ay1-ay)*dt

    t += dt

    return x,y,vx,vy,t

# Set initial position and velocity.

x0 = 0.0
y0 = 0.0

v0    = 100.0				# unit: m/s 
theta = 60.0				# (degrees) , angle with horizontal 

# Determine components of the initial velocity vector.
vx0 = v0 * math.cos( math.radians(theta) )
vy0 = v0 * math.sin( math.radians(theta) )

# Set parameters governing the numerical details.
dt    = 0.1
t_max = 20.0

# Initialize the trajectory.
t = 0.0
tp = t
tpp = tp
x = x0
xp = x
xpp = xp
y = y0
yp = y
ypp = yp
vx = vx0
vy = vy0

tplot = []
xplot = []
yplot = []

#print(x, y, t)

# Calculate the trajectory to time t_max, using the 2D
# predictor-corrector scheme.

while y >= 0:
    xpp = xp
    tpp = tp
    ypp = yp
    xp = x
    tp = t
    yp = y
    x,y,vx,vy,t = take_a_step(x, y, vx, vy, t, dt)
    
    #if ypp < yp and yp > y:
        #print("Numerical Height =", interp(xpp, ypp, xp, yp, x))
    
    
    tplot.append(t)
    xplot.append(x)
    yplot.append(y)
    #print(x, y, t)

theta2 = math.atan(vy/vx)
print('Initial angle =', theta)
print('Angle from horozontal which projectile strikes the ground =', abs(math.degrees(theta2)))
print('\nRange =',interp(ypp, xpp, yp, xp, 0))
print('Time in Flight =', interp(ypp, tpp, yp, tp, 0))
#print ('New velocity =', v0)
plt.plot(xplot,yplot)
plt.title('Projectile path for beta = 0.001' )
plt.xlabel('x')
plt.ylabel('y')
plt.show()

v0 = 1500

print('Angle from horozontal which projectile strikes the ground when there is no air resistance = 60.07286807603103')
print('Range when there is no air resistance = 883.7363277282706')


# In[18]:


#part c
import math
import matplotlib.pyplot as plt


# acceleration due to gravity
g = 9.8
beta = 0.001  #<=== Beta

def accx(x, y, vx, vy, t):
    return  - beta * math.sqrt((vx**2)+(vy**2)) * vx

def accy(x, y, vx, vy, t):
    return  -g - beta * math.sqrt((vx**2)+(vy**2)) * vy

def interp(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def take_a_step(x, y, vx, vy, t, dt):

    ax = accx(x, y, vx, vy, t)
    ay = accy(x, y, vx, vy, t)

    # Predict: 
    x  += vx*dt + 0.5*ax*dt*dt
    y  += vy*dt + 0.5*ay*dt*dt
    vx += ax*dt
    vy += ay*dt

    # Correct: 
    ax1 = accx(x, y, vx, vy, t)
    ay1 = accy(x, y, vx, vy, t)

    vx += 0.5*(ax1-ax)*dt
    vy += 0.5*(ay1-ay)*dt

    t += dt

    return x,y,vx,vy,t

# Set initial position and velocity.

x0 = 0.0
y0 = 0.0

v0    = 176.0				# unit: m/s 
theta = 60.0				# (degrees) , angle with horizontal 

# Determine components of the initial velocity vector.
vx0 = v0 * math.cos( math.radians(theta) )
vy0 = v0 * math.sin( math.radians(theta) )

# Set parameters governing the numerical details.
dt    = 0.1
t_max = 20.0

# Initialize the trajectory.
t = 0.0
tp = t
tpp = tp
x = x0
xp = x
xpp = xp
y = y0
yp = y
ypp = yp
vx = vx0
vy = vy0

tplot = []
xplot = []
yplot = []

#print(x, y, t)

# Calculate the trajectory to time t_max, using the 2D
# predictor-corrector scheme.

while y >= 0:
    xpp = xp
    tpp = tp
    ypp = yp
    xp = x
    tp = t
    yp = y
    x,y,vx,vy,t = take_a_step(x, y, vx, vy, t, dt)
    
    #if ypp < yp and yp > y:
        #print("Numerical Height =", interp(xpp, ypp, xp, yp, x))
    
    
    tplot.append(t)
    xplot.append(x)
    yplot.append(y)
    #print(x, y, t)

theta2 = math.atan(vy/vx)
#print('Initial angle =', theta)
#print('Angle from horozontal which projectile strikes the ground =', abs(math.degrees(theta2)))
print('\nRange =',interp(ypp, xpp, yp, xp, 0))
#print('Time in Flight =', interp(ypp, tpp, yp, tp, 0))
print ('New velocity =', v0)
plt.plot(xplot,yplot)
plt.title('Projectile path for beta = 0.001' )
plt.xlabel('x')
plt.ylabel('y')
plt.show()

v0 = 100

#print('Angle from horozontal which projectile strikes the ground when there is no air resistance = 60.07286807603103')
print('Range when there is no air resistance = 883.7363277282706')


#print("Analytical Height = ", .5*((v0**2)/g)*math.sin(math.radians(theta))**2)
print('Old Velocity =', v0)
#print("Analytical Range = ", ((2*(v0**2))/g)*math.cos(math.radians(theta))*math.sin(math.radians(theta)))
#print("Analytical (old) Time in Flight = ", ((2*(v0))/g)*math.sin(math.radians(theta)))  

print('\nDecrease in Range =', (883.7363277282706 - interp(ypp, xpp, yp, xp, 0)))
print('% decrease = ', 883.7363277282706 / interp(ypp, xpp, yp, xp, 0) )


# In[44]:


# part d
import math
import matplotlib.pyplot as plt


# acceleration due to gravity
g = 9.8
beta = 0.001  #<=== Beta

def accx(x, y, vx, vy, t):
    return  - beta * math.sqrt((vx**2)+(vy**2)) * vx

def accy(x, y, vx, vy, t):
    return  -g - beta * math.sqrt((vx**2)+(vy**2)) * vy

def interp(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def take_a_step(x, y, vx, vy, t, dt):

    ax = accx(x, y, vx, vy, t)
    ay = accy(x, y, vx, vy, t)

    # Predict: 
    x  += vx*dt + 0.5*ax*dt*dt
    y  += vy*dt + 0.5*ay*dt*dt
    vx += ax*dt
    vy += ay*dt

    # Correct: 
    ax1 = accx(x, y, vx, vy, t)
    ay1 = accy(x, y, vx, vy, t)

    vx += 0.5*(ax1-ax)*dt
    vy += 0.5*(ay1-ay)*dt

    t += dt

    return x,y,vx,vy,t

# Set initial position and velocity.

x0 = 0.0
y0 = 0.0

v0    = 100.0				# unit: m/s 
theta = 41.0				# (degrees) , angle with horizontal, found by trial and error

# Determine components of the initial velocity vector.
vx0 = v0 * math.cos( math.radians(theta) )
vy0 = v0 * math.sin( math.radians(theta) )

# Set parameters governing the numerical details.
dt    = 0.1
t_max = 20.0

# Initialize the trajectory.
t = 0.0
tp = t
tpp = tp
x = x0
xp = x
xpp = xp
y = y0
yp = y
ypp = yp
vx = vx0
vy = vy0

tplot = []
xplot = []
yplot = []

#print(x, y, t)

# Calculate the trajectory to time t_max, using the 2D
# predictor-corrector scheme.

while y >= 0:
    xpp = xp
    tpp = tp
    ypp = yp
    xp = x
    tp = t
    yp = y
    x,y,vx,vy,t = take_a_step(x, y, vx, vy, t, dt)
    
    #if ypp < yp and yp > y:
        #print("Numerical Height =", interp(xpp, ypp, xp, yp, x))
    
    
    tplot.append(t)
    xplot.append(x)
    yplot.append(y)
    #print(x, y, t)

theta2 = math.atan(vy/vx)
print('Theta Naught =', theta)
print('Theta One =', abs(math.degrees(theta2)))
print('\nRange =',interp(ypp, xpp, yp, xp, 0))

print('Range (when theta naught=60) = 500.46606048056356')
#print('Time in Flight =', interp(ypp, tpp, yp, tp, 0))
#print ('New velocity =', v0)
plt.plot(xplot,yplot)
plt.title('Projectile path for beta = 0.001' )
plt.xlabel('x')
plt.ylabel('y')
plt.show()

v0 = 100

#print('Angle from horozontal which projectile strikes the ground when there is no air resistance = 60.07286807603103')
#print('Range when there is no air resistance = 883.7363277282706')


#print("Analytical Height = ", .5*((v0**2)/g)*math.sin(math.radians(theta))**2)
#print('Old Velocity =', v0)
#print("Analytical Range = ", ((2*(v0**2))/g)*math.cos(math.radians(theta))*math.sin(math.radians(theta)))
#print("Analytical (old) Time in Flight = ", ((2*(v0))/g)*math.sin(math.radians(theta)))  

#print('\nDecrease in Range =', (883.7363277282706 - interp(ypp, xpp, yp, xp, 0)))
#print('% decrease = ', 883.7363277282706 / interp(ypp, xpp, yp, xp, 0) )


# In[51]:


#Part e

import math
import matplotlib.pyplot as plt


# acceleration due to gravity
g = 9.8
beta = 0.001  #<=== Beta

def accx(x, y, vx, vy, t):
    return  - beta * math.sqrt((vx**2)+(vy**2)) * vx

def accy(x, y, vx, vy, t):
    return  -g - beta * math.sqrt((vx**2)+(vy**2)) * vy

def interp(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def take_a_step(x, y, vx, vy, t, dt):

    ax = accx(x, y, vx, vy, t)
    ay = accy(x, y, vx, vy, t)

    # Predict: 
    x  += vx*dt + 0.5*ax*dt*dt
    y  += vy*dt + 0.5*ay*dt*dt
    vx += ax*dt
    vy += ay*dt

    # Correct: 
    ax1 = accx(x, y, vx, vy, t)
    ay1 = accy(x, y, vx, vy, t)

    vx += 0.5*(ax1-ax)*dt
    vy += 0.5*(ay1-ay)*dt

    t += dt

    return x,y,vx,vy,t

# Set initial position and velocity.

x0 = 0.0
y0 = 0.0

v0    = 100.0				# unit: m/s 
theta = 0.0				# (degrees) , angle with horizontal 

thetanaught = []
thetaone = []

while theta <= 90:
    # Determine components of the initial velocity vector.
    vx0 = v0 * math.cos( math.radians(theta) )
    vy0 = v0 * math.sin( math.radians(theta) )

    # Set parameters governing the numerical details.
    dt    = 0.1
    t_max = 20.0

    # Initialize the trajectory.
    t = 0.0
    tp = t
    tpp = tp
    x = x0
    xp = x
    xpp = xp
    y = y0
    yp = y
    ypp = yp
    vx = vx0
    vy = vy0

    tplot = []
    xplot = []
    yplot = []

    #print(x, y, t)

    # Calculate the trajectory to time t_max, using the 2D
    # predictor-corrector scheme.

    while y >= 0:
        xpp = xp
        tpp = tp
        ypp = yp
        xp = x
        tp = t
        yp = y
        x,y,vx,vy,t = take_a_step(x, y, vx, vy, t, dt)

        #if ypp < yp and yp > y:
            #print("Numerical Height =", interp(xpp, ypp, xp, yp, x))


        tplot.append(t)
        xplot.append(x)
        yplot.append(y)
        #print(x, y, t)

    theta2 = math.atan(vy/vx)
    thetanaught.append(theta)
    thetaone.append(abs(math.degrees(theta2)))
    
    theta = theta + 1
    #print('\nRange =',interp(ypp, xpp, yp, xp, 0))

#print('Range (when theta naught=60) = 500.46606048056356')
#print('Time in Flight =', interp(ypp, tpp, yp, tp, 0))
#print ('New velocity =', v0)
plt.plot(thetanaught, thetaone)
plt.title('Theta One as a function of Theta Naught' )
plt.xlabel('Theta naught')
plt.ylabel('Theta One')
plt.show()


# In[ ]:




