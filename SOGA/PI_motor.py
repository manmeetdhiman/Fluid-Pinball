#PI Controller and DC Motor Model

def PI_motor(w_des,dt,tsteps):
    # imports
    import numpy as np
    #import matplotlib.pyplot as plt
    from scipy.integrate import odeint

    # DC Motor Control  using PI controller

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

    #PID parameters
    error_s = np.zeros(tsteps)
    V_bias = 0
    tau_i = 30
    sum_int = 0.0

    #Tunable parameters
    Kp = 4
    Ki = Kp / tau_i

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
        V_PID = V_bias + Kp * error + Ki * sum_int

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


    '''# plotting
    fig = plt.subplots(2, 1, constrained_layout=False)

    # PID input
    plt.subplot(2, 2, 1)
    plt.plot(t_sim, w_des * 9.5493)
    plt.title('Desired Angular Velocity')
    plt.ylabel('Angular velocity (RPM)')
    plt.xlabel('time(s)')

    # error
    plt.subplot(2, 2, 2)
    plt.plot(t_sim, error_s)
    plt.title('Error')
    plt.ylabel('error')
    plt.xlabel('time(s)')

    # DC Motor input
    plt.subplot(2, 2, 3)
    plt.plot(t_sim, V_in)
    plt.title('PI Voltage Output')
    plt.ylabel('Voltage Input (V)')
    plt.xlabel('time(s)')

    # DC Motor response
    plt.subplot(2, 2, 4)
    plt.plot(t_sim, w_s * 9.5493)
    plt.title('DC Motor Response')
    plt.ylabel('Angular Velocity (RPM)')
    plt.xlabel('time (s)')

    plt.show()'''
    return w_s





