import numpy as np
import matplotlib.pyplot as plt

def main():
    # Define initial conditions and parameters
    y0 = 0
    t0 = 0
    tf = np.pi/2
    h = 0.1

    # Euler's method
    t_euler, y_euler = Euler(f, y0, t0, tf, h)

    # RK4 method
    t_rk4, y_rk4 = RK4(f, y0, t0, tf, h)

    t_true = np.linspace(t0, tf, 100)
    y_true = np.tan(t_true)

    # Label intervals of pi/2
    labels = [r'$0$', r'$\pi/4$', r'$\pi/2$']
    positions = np.arange(t0, np.pi/2 + np.pi/4, np.pi/4)

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



# f if the first order ODE in the form of 
#        dy/dt = f(t, y)
# (y0, t0) --- initial condition
# tf       --- final time
# h        --- time step
def Euler(f, y0, t0, tf, h):

    # Defines the time array
    t = np.arange(t0, tf+h, h)
    
    # Initialize the solution array
    y = np.zeros_like(t)

    # Set initial condition
    y[0] = y0            

    # Euler's method
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(t[i-1], y[i-1])

    return t, y

def RK4(f, y0, t0, tf, h):

    # Defines the time array
    t = np.arange(t0, tf+h, h)
    
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
    return y*y + 1

if __name__ == "__main__":
    main()