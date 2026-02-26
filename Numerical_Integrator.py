import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Define initial conditions and parameters
    y0 = 0
    t0 = 0
    tf = 3*np.pi/4
    h = 0.1

    # Euler's method
    t_euler, y_euler = Euler(f, y0, t0, tf, h)

    # RK4 method
    t_rk4, y_rk4 = RK4(f, y0, t0, tf, h)

    t_true = np.linspace(t0, tf, 100)
    y_true = np.tan(t_true)

    # Label intervals of pi/2
    labels = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$']
    positions = np.arange(t0, 3*np.pi/4 + np.pi/4, np.pi/4)

    # Print results
    plt.figure
    plt.plot(t_euler, y_euler, label='Euler Method',
                marker = 'o', linestyle = '-.')
    plt.plot(t_rk4, y_rk4, label='RK4 Method',
                marker='x', linestyle='--')
    plt.plot(t_true, y_true, label='True Solution', linestyle='-')
    plt.xlabel('Time (t)')
    plt.ylabel('y(t)')
    plt.title('Numerical Integration of ODE')
    plt.legend()
    plt.xticks(positions, labels)
    plt.ylim(0, 100)
    plt.grid()
    plt.show()

    # Convergence Study 
    y0 = 0
    t0 = 0
    tf = 0.5
    y_true = np.tan(tf)
    h_values = np.array([0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001])

    # Global accumulated error
    y_global_euler = np.zeros_like(h_values)
    y_global_rk4   = np.zeros_like(h_values)

    # Enumerate h values
    for i, h in enumerate(h_values):
        t_euler, y_euler = Euler(f, y0, t0, tf, h)
        t_rk4, y_rk4 = RK4(f, y0, t0, tf, h)

        # Get last value of the numerical solution
        y_global_euler[i] = y_euler[-1]
        y_global_rk4[i]   = y_rk4[-1]

    # Find Global errors
    length = len(h_values)
    errors_euler = np.zeros(length)
    errors_rk4   = np.zeros(length)

    for i in range(length):
        errors_euler[i] = abs(y_global_euler[-1] - y_global_euler[i])
        errors_rk4[i]   = abs(y_global_rk4[-1]   - y_global_rk4[i])

    # Plot convergence
    plt.figure()
    plt.loglog(1/h_values[0:length-1], errors_euler[0:length-1], label='Euler Error', marker='o')
    plt.loglog(1/h_values[0:length-1], errors_rk4[0:length-1]  , label='RK4 Error'  , marker='x')
    plt.xlabel('Time Step (1/h)')
    plt.ylabel('Global Error')
    plt.title('Convergence Study')
    plt.legend()
    plt.grid()
    plt.show()

    # Make Table of results
    data = {
        'h': h_values,
        'Euler Error': errors_euler,
        'RK4 Error': errors_rk4
    }
    df = pd.DataFrame(data)
    print(df)

    # Compute slopes
    fit_range = np.arange(3, 7)   # adjust as needed

    slope_euler = np.polyfit(np.log(h_values[fit_range]),
                         np.log(errors_euler[fit_range]), 1)[0]

    slope_rk4 = np.polyfit(np.log(h_values[fit_range]),
                       np.log(errors_rk4[fit_range]), 1)[0]

    # Print to Screen
    print("Euler order ≈", slope_euler)
    print("RK4 order ≈", slope_rk4)

# f if the first order ODE in the form of 
#        dy/dt = f(t, y)
# (y0, t0) --- initial condition
# tf       --- final time
# h        --- time step
def Euler(f, y0, t0, tf, h):

    # Number of steps
    N = int((tf - t0)/h)

    # Defines the time array
    t = np.linspace(t0, tf, N+1)
    
    # Initialize the solution array
    y = np.zeros_like(t)

    # Set initial condition
    y[0] = y0            

    # Euler's method
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(t[i-1], y[i-1])

    return t, y

def RK4(f, y0, t0, tf, h):

    # Number of steps
    N = int((tf - t0)/h)

    # Defines the time array
    t = np.linspace(t0, tf, N+1)

    # Initialize the solution array
    y = np.zeros_like(t)

    # Set initial condition
    y[0] = y0

    # RK4 Algorithm
    for i in range(1, len(t)):
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + h/2, y[i-1] + k1*h/2)
        k3 = f(t[i-1] + h/2, y[i-1] + k2*h/2)
        k4 = f(t[i-1] + h  , y[i-1] + k3*h  )

        y[i] = y[i-1] + h*(k1 + 2*k2 + 2*k3 + k4)/6

    return t, y

def f(t, y):
    return y**2 + 1

if __name__ == "__main__":
    main()