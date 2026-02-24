import numpy as np
import matplotlib.pyplot as plt

def main():
    # Temperature of surface of star
    T = 10000

    # Velecity range from 0 to 100,000 m/s
    v = np.linspace(0, 50000, 1000)

    # Mass of hydrogen atom in kg
    m = 1.67e-27

    # Plotting the Maxwell-Boltzmann distribution
    plt.figure()
    plt.plot(v, Maxwell_Boltzmann_Distribution(v, T, 1.67e-27), label='Maxwell-Boltzmann Distribution')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Probability Density')
    plt.title('Maxwell-Boltzmann Distribution for Hydrogen at 10000 K')
    plt.legend()
    plt.show()


# T --- Tempurature in Kelvin
# m --- Mass of the particle in kg
# v --- Speed of the particle in m/s
def Maxwell_Boltzmann_Distribution(v, T, m):

    # Boltzmann constant in J/K
    k_B = 1.380649e-23 
    
    # Maxwell-Boltzmann distribution formula 
    f = (m / (2 * np.pi * k_B * T))**(3/2) * 4 * np.pi * v**2 * np.exp(-m * v**2 / (2 * k_B * T))
    
    return f

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