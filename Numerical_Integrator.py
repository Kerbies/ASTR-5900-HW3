import numpy as np

def main():
    pass

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

if __name__ == "__main__":
    main()