#Motor Model Only For Analysis
#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



#Define input parameters here
V_input = 36  #step input voltage
dt = 5e-4
tsteps = 100


#Simulation
# Motor Parameters
Ke = 0.021199438  # V/rad/s
Kt = 0.0141937  # Nm/A
b = 0.0001011492  # Nm/rad/s
L = 0.00075  # H
J = 0.00000109445  # kgm^2
R = 1.56  # ohms
V_max = 36
V_min = -V_max

# intial conditions
i0 = 0
w0 = 0

# simulation time parameters
tf = dt * (tsteps - 1)
t_sim = np.linspace(0, tf, tsteps)

# Motor input
V_in = np.zeros(tsteps)
V_in[11:] = V_input

# ODE Output Storage
w_s = np.zeros(tsteps)
i_s = np.zeros(tsteps)

# DC Motor Model ODEs
def motor_electrical(i, t, V, w):
    di_dt = (V - R * i - Ke * w) / L
    return di_dt

def motor_mechanical(w, t, i):
    dw_dt = (Kt * i - b * w) / J
    return dw_dt

# # Motor actuation sim_loop
for n in range(tsteps - 1):

    V = V_in[n]
    t_range = [t_sim[n], t_sim[n + 1]]

    i = odeint(motor_electrical, i0, t_range, args=(V, w0))
    i0 = i[-1][0]
    i_s[n + 1] = i0
    w = odeint(motor_mechanical, w0, t_range, args=(i0,))
    w0 = w[-1]
    w_s[n + 1] = w0


# plotting
fig = plt.subplots(2, 1, constrained_layout=False)

# DC Motor input
plt.subplot(3, 1, 1)
plt.plot(t_sim, V_in)
plt.title('PI Voltage Output')
plt.ylabel('Voltage Input (V)')
plt.xlabel('time(s)')


# DC Motor response (RPMs)
plt.subplot(3, 1, 2)
plt.plot(t_sim, w_s)
plt.title('DC Motor Response (rad/s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.xlabel('time (s)')

# DC Motor response (RPMs)
plt.subplot(3, 1, 3)
plt.plot(t_sim, w_s * 9.5493)
plt.title('DC Motor Response (RPMs)')
plt.ylabel('Angular Velocity (RPMs)')
plt.xlabel('time (s)')


plt.show()



