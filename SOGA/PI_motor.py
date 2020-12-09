#PI Controller and DC Motor Model Only for analysis
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

w_max = 1047.1975513824 #rad/s
f = 16#Hz
#Define input parameters here
A = w_max
omega = f*6.28
dt = 5e-4
tsteps = 100

# simulation time parameters
tf = dt * (tsteps - 1)
t_sim = np.linspace(0, tf, tsteps)

w_desired =  np.zeros(tsteps) #rad/s   Step input angular velocity

for n in range(len(t_sim)):
    w_desired[n] = A*math.sin(omega*t_sim[n])


#Simulation
# Motor Parameters
Ke = 0.021199438  # V/rad/s
Kt = 0.0141937  # Nm/A
b = 0.0001011492  # Nm/rad/s
L = 0.00075  # H
J = 0.00000109445# kgm^2
R = 1.56  # ohms
V_max = 36
V_min = -V_max

#intial conditions
i0 = 0
w0 = 0


#PID parameters
error_s = np.zeros(tsteps)
V_bias = 0
sum_int = 0.0

#Tunable parameters
Kp = 0.270727147578817
Ki = 50.0897752327866
Kd = 0.000141076220179068
N = 248711.202620588 #Filter coefficient for filtered derivative
#PI Input
w_des = np.zeros(tsteps)
w_des = w_desired

# Motor input
V_in = np.zeros(tsteps)


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

# sim_loop
for n in range(tsteps - 1):

    # PID Control
    error = w_des[n + 1] - w0
    error_s[n + 1] = error

    sum_int = sum_int + error * dt
    de_dt = (error_s[n+1] - error_s[n])/dt
    V_PID = V_bias + Kp * error + Ki * sum_int #+ (N*Kd)/(1 + N*sum_int) #+ Kd*de_dt

    # anti-integral windup
    if V_PID > V_max:
        V_PID = V_max
        sum_int = sum_int - error * dt
    if V_PID < V_min:
        V_PID = V_min
        sum_int = sum_int - error * dt

    # PID Data storage
    int_s = sum_int
    V_in[n] = V_PID
    # Motor Actuation
    V = V_in[n]
    t_range = [t_sim[n], t_sim[n + 1]]

    i = odeint(motor_electrical, i0, t_range, args=(V, w0))
    i0 = i[-1][0]
    i_s[n + 1] = i0
    w = odeint(motor_mechanical, w0, t_range, args=(i0,))
    w0 = w[-1]
    w_s[n + 1] = w0

#response charachteristics
'''response_error = ((w_s[-1] - w_desired)/w_desired) *100
p_overshoot = ((max(w_s) - w_desired)/w_desired) *100

print(f'response_error is: {response_error} %' )
if p_overshoot < 0:
    print(f'No overshoot')
else:
    print(f'percent overshoot is: {p_overshoot} %')
# plotting
fig = plt.subplots(2, 1, constrained_layout=False)'''

# DC Motor command and response
plt.subplot(2, 1, 1)
plt.plot(t_sim, w_s,t_sim,w_des)
plt.title('DC Motor Response')
plt.ylabel('Angular Velocity (rad/s)')
plt.xlabel('time (s)')


# DC Motor input
plt.subplot(2, 1, 2)
plt.plot(t_sim, V_in)
plt.title('PI Voltage Output')
plt.ylabel('Voltage Input (V)')
plt.xlabel('time(s)')



plt.show()


