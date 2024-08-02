import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use('TkAgg')

# Konstanty
g = 9.81  # gravitační zrychlení (m/s^2)
m1 = 1.0  # hmotnost 1. kyvadla (kg)
m2 = 1.0  # hmotnost 2. kyvadla (kg)
L1 = 1.0  # délka 1. kyvadla (m)
L2 = 1.0  # délka 2. kyvadla (m)

# Funkce pro kinetickou energii
def kinetic_energy(theta1_dot, theta2_dot, delta_theta):
    T = 0.5 * m1 * (L1**2) * (theta1_dot**2) + \
        0.5 * m2 * (L1**2 * (theta1_dot**2) + L2**2 * (theta2_dot**2) + \
        2 * L1 * L2 * theta1_dot * theta2_dot * np.cos(delta_theta))
    return T

# Funkce pro potenciální energii
def potential_energy(theta1, theta2):
    V = m1 * g * L1 * (1 - np.cos(theta1)) + \
        m2 * g * (L1 * (1 - np.cos(theta1)) + L2 * (1 - np.cos(theta2)))
    return V

# Funkce pro výpočet změny úhlů a úhlových rychlostí
def derivatives(state):
    theta1, omega1, theta2, omega2 = state
    delta_theta = theta1 - theta2
    A = m1 + m2 * np.sin(delta_theta)**2
    
    d_omega1 = (-np.sin(delta_theta) * (m2 * L1 * omega1**2 * np.cos(delta_theta) + m2 * L2 * omega2**2) - \
                g * (m1 * np.sin(theta1) - m2 * np.sin(theta2) * np.cos(delta_theta))) / (2 * L1 * A)
    
    d_omega2 = (+np.sin(delta_theta) * (m2 * L2 * omega2**2 * np.cos(delta_theta) + m1 * L1 * omega1**2) - \
                g * (m1 * np.sin(theta2) - m1 * np.sin(theta1) * np.cos(delta_theta))) / (2 * L2 * A)
    
    d_theta1 = omega1
    d_theta2 = omega2
    
    return np.array([d_theta1, d_omega1, d_theta2, d_omega2])

# Runge-Kuttova metoda 4. řádu
def runge_kutta_4th_order(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state

# Generování počátečních podmínek
def generate_initial_conditions():
    while True:
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(-np.pi, np.pi)
        
        V = potential_energy(theta1, theta2)
        E = V + 1e-5  # Zajištění, že E je mírně větší než V

        delta_theta = theta1 - theta2
        max_T = E - V
        
        if max_T < 0:
            continue
        
        max_omega1 = np.sqrt(max_T / (0.5 * m1 * L1**2))
        omega1 = np.random.uniform(-max_omega1, max_omega1)
        
        T_remaining = max_T - 0.5 * m1 * L1**2 * omega1**2
        if T_remaining < 0:
            continue
        
        omega2 = np.sqrt(T_remaining / (0.5 * m2 * L2**2))
        
        if kinetic_energy(omega1, omega2, delta_theta) + V <= E:
            return np.array([theta1, omega1, theta2, omega2]), E

# Hlavní funkce pro simulaci
def simulate_pendulum(t_max, dt):
    state, E0 = generate_initial_conditions()
    
    t = 0
    times = []
    theta1_vals = []
    theta2_vals = []
    energies = []
    
    while t < t_max:
        times.append(t)
        theta1_vals.append(state[0])
        theta2_vals.append(state[2])
        
        # Výpočet celkové energie
        V = potential_energy(state[0], state[2])
        T = kinetic_energy(state[1], state[3], state[0] - state[2])
        E = T + V
        
        # Oprava energie, aby byla co nejblíže k E0
        energy_correction = E0 - E
        if np.abs(energy_correction) > 1e-6:
            E = E0  # Oprava energie na E0
        
        energies.append(E)
        
        # Aplikace Runge-Kuttovy metody
        state = runge_kutta_4th_order(state, dt)
        
        t += dt
    
    # Grafy výsledků
    plt.figure(figsize=(12, 6))

    # Graf úhlů
    plt.subplot(2, 1, 1)
    plt.plot(times, theta1_vals, label=r'$\theta_1$', color='blue')
    plt.plot(times, theta2_vals, label=r'$\theta_2$', color='red')
    plt.xlabel('Čas (s)')
    plt.ylabel('Úhel (rad)')
    plt.title('Úhly kyvadel')
    plt.legend()

    # Graf energie
    plt.subplot(2, 1, 2)
    plt.plot(times, energies, label='Energie', color='green')
    plt.xlabel('Čas (s)')
    plt.ylabel('Energie (J)')
    plt.title('Celková energie')
    plt.axhline(y=E0, color='r', linestyle='--', label='Počáteční energie')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Funkce pro animaci
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate_pendulum(i):
        global state
        state = runge_kutta_4th_order(state, dt)
        theta1, _, theta2, _ = state

        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)

        line.set_data([0, x1, x2], [0, y1, y2])
        time_text.set_text(time_template % (i * dt))
        return line, time_text
    
    # Nastavení animace
    fig, ax = plt.subplots()
    ax.set_xlim(-2 * (L1 + L2), 2 * (L1 + L2))
    ax.set_ylim(-2 * (L1 + L2), 2 * (L1 + L2))
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    line, = ax.plot([], [], 'o-', lw=2, color='blue')
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    ani = FuncAnimation(fig, animate_pendulum, frames=int(t_max / dt), init_func=init, blit=True, repeat=False)
    
    # Zobrazení animace
    plt.show()
    plt.ion()
    ani
# Spuštění simulace
simulate_pendulum(t_max=20, dt=0.1)



