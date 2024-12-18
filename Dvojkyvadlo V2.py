import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Konstanty
g = 9.81  # gravitační zrychlení v m/s^2
L1 = 1.0  # délka prvního kyvadla v m
L2 = 1.0  # délka druhého kyvadla v m
m1 = 1.0  # hmotnost prvního závaží v kg
m2 = 1.0  # hmotnost druhého závažív kg

# Funkce na počítání celkové energie systému
def total_energy(theta1, omega1, theta2, omega2):
    potential_energy = m1 * g * L1 * (1 - np.cos(theta1)) + m2 * g * (L1 * (1 - np.cos(theta1)) + L2 * (1 - np.cos(theta2)))
    kinetic_energy = 0.5 * m1 * (L1 * omega1)**2 + 0.5 * m2 * ((L1 * omega1)**2 + (L2 * omega2)**2 +
                                                                2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2))
    return potential_energy + kinetic_energy

# Funkce na generování náhodných počátečních podmínek pro danou energii
def generate_initial_conditions(energy):
    while True:
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(-np.pi, np.pi)
        omega1 = np.random.uniform(-10, 10)
        omega2 = np.random.uniform(-10, 10)
        if np.abs(total_energy(theta1, omega1, theta2, omega2) - energy) < 0.1:
            return [theta1, omega1, theta2, omega2]

# Časové parametry
dt = 0.1  # časový krok v s
t_max = 20  # celkový čas simulace v s

# Seznamy na data Poincarého řezu
poincare_theta2 = []
poincare_dtheta2 = []

# Nastavení parametrů
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_xlim(-2.2, 2.2)
ax1.set_ylim(-2.2, 2.2)
line, = ax1.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

ax2.set_xlim(-np.pi, np.pi)
ax2.set_ylim(-10, 10)
ax2.set_xlabel(r'$\theta_2$')
ax2.set_ylabel(r'$\dot{\theta}_2$')
ax2.set_title('Poincarého řez')
poincare_points, = ax2.plot([], [], 'ro', markersize=2)

# Spuštění animace
def init():
    line.set_data([], [])
    time_text.set_text('')
    poincare_points.set_data([], [])
    return line, time_text, poincare_points

previous_theta1 = None

# Normalizace úhlů na interval [-pi, pi]
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Animační funkce
def animate(i):
    global state, previous_theta1
    state = position_verlet(state, dt)
    theta1, omega1, theta2, omega2 = state

    theta1 = normalize_angle(theta1)
    theta2 = normalize_angle(theta2)

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    line.set_data([0, x1, x2], [0, y1, y2])
    time_text.set_text(time_template % (i * dt))

    # Kontrola pro protnutí s rovinou theta1 = 0
    if previous_theta1 is not None:
        normalized_previous_theta1 = normalize_angle(previous_theta1)
        if normalized_previous_theta1 * theta1 < 0:  # theta1 protne 0
            poincare_theta2.append(theta2)
            poincare_dtheta2.append(omega2)
            poincare_points.set_data(poincare_theta2, poincare_dtheta2)

    previous_theta1 = theta1
    
    print(total_energy(theta1, omega1, theta2, omega2))
    
    return line, time_text, poincare_points


plt.ion()

# Počáteční energie
Energy = float(input("Vložte požadovanou celkovou energii systému: "))

# Generování náhodných počátečních podmínek pro danou energii
state = generate_initial_conditions(Energy)

# Verletova integrační metoda
def position_verlet(state, dt):
    theta1, omega1, theta2, omega2 = state
    # Rychlosti o půl kroku
    omega1_half = omega1 + 0.5 * dt * (-g/L1) * np.sin(theta1)
    omega2_half = omega2 + 0.5 * dt * (-g/L2) * np.sin(theta2)

    # Aktualizace poloh
    theta1 += omega1_half * dt
    theta2 += omega2_half * dt

    #  Rychlosti o jeden krok
    omega1 = omega1_half + 0.5 * dt * (-g/L1) * np.sin(theta1)
    omega2 = omega2_half + 0.5 * dt * (-g/L2) * np.sin(theta2)

    return [theta1, omega1, theta2, omega2]


ani = FuncAnimation(fig, animate, frames=int(t_max / dt), init_func=init, blit=True)


plt.show()


ani



