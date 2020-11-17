#!/usr/bin/env python
# coding: utf-8

# - Dmitri LaBelle 
# - Hw 6 Problem 2 
# - The purpose of this code is to consider a projectile system where air resistance varies with height of the projectile, even more similar to natural observation. From here, and with a given h value (which is 500m in part A and 5000m in part B), we repeat the experiement done in problem 1 part d, finding the maximum range given a single starting velocity and varying take off angle. 
# - A problem I had with this code as well as 1.d was the efficency and accuracy of trial and error. While I was somewhat more strenuous in testing, it probably would have been easier to apply a nested while loop over a theta minimum and maximum to find the max range. This technique will be used later on in problem 3 part b. 

# In[2]:


#part a
import math
import matplotlib.pyplot as plt


# acceleration due to gravity
g = 9.8
beta = 0 #0.001  #<=== Beta
h = 500

def beta(y):
    return 0.001 * math.exp(-y/h)

def accx(x, y, vx, vy, t):
    return  - beta(y) * math.sqrt((vx**2)+(vy**2)) * vx

def accy(x, y, vx, vy, t):
    return  -g - beta(y) * math.sqrt((vx**2)+(vy**2)) * vy

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
theta = 43.5				# (degrees) , angle with horizontal, found by trial and error

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

#theta2 = math.atan(vy/vx)
#print('Initial angle =', theta)
#print('Angle from horozontal which projectile strikes the ground =', abs(math.degrees(theta2)))
print('Theta =', theta)
print('\nRange =',interp(ypp, xpp, yp, xp, 0))
print('Range when theta=60 = 558.0480112439722',)
#print('Time in Flight =', interp(ypp, tpp, yp, tp, 0))
#print ('New velocity =', v0)
plt.plot(xplot,yplot)
plt.title('Projectile path for beta = 0.001*e^-y/h' )
plt.xlabel('x')
plt.ylabel('y')
plt.show()

v0 = 1500

#print('Angle from horozontal which projectile strikes the ground when there is no air resistance = 60.07286807603103')
#print('Range when there is no air resistance = 883.7363277282706')


# In[37]:


#part b
import math
import matplotlib.pyplot as plt


# acceleration due to gravity
g = 9.8
beta = 0 #0.001  #<=== Beta
h = 5000

def beta(y):
    return 0.001 * math.exp(-y/h)

def accx(x, y, vx, vy, t):
    return  - beta(y) * math.sqrt((vx**2)+(vy**2)) * vx

def accy(x, y, vx, vy, t):
    return  -g - beta(y) * math.sqrt((vx**2)+(vy**2)) * vy

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

#theta2 = math.atan(vy/vx)
#print('Initial angle =', theta)
#print('Angle from horozontal which projectile strikes the ground =', abs(math.degrees(theta2)))
print('Theta =', theta)
print('Range =',interp(ypp, xpp, yp, xp, 0))
print('Range when theta=60 = 506.31634176381664',)
#print('Time in Flight =', interp(ypp, tpp, yp, tp, 0))
#print ('New velocity =', v0)
plt.plot(xplot,yplot)
plt.title('Projectile path for beta = 0.001*e^-y/h' )
plt.xlabel('x')
plt.ylabel('y')
plt.show()

v0 = 1500

#print('Angle from horozontal which projectile strikes the ground when there is no air resistance = 60.07286807603103')
#print('Range when there is no air resistance = 883.7363277282706')


# In[ ]:




