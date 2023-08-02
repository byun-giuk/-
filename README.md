# -
수학 실생활
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Satellite:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity

def compute_gravitational_force(position, mass):
    G = 6.67430e-11
    earth_mass = 5.972e24
    distance = np.linalg.norm(position)
    direction = -position / distance
    force_magnitude = G*mass*earth_mass / (distance**2)
    return force_magnitude * direction

def update_satellite(sat, timestep, mass):
    force = compute_gravitational_force(sat.position, mass)
    acceleration = force / mass
    sat.velocity += acceleration * timestep
    sat.position += sat.velocity * timestep

def simulate_satellite_orbits(satellites, timestep, steps, mass):
    orbits = np.empty((steps, len(satellites), 3))
    for i in range(steps):
        for n, sat in enumerate(satellites):
            update_satellite(sat, timestep, mass)
            orbits[i][n] = sat.position
    return orbits

def plot_orbits(orbits):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    colors = ["r", "g", "b", "y"]
    
    for n, satellite_positions in enumerate(orbits.T):
        ax.plot(satellite_positions[:, 0], satellite_positions[:, 1], satellite_positions[:, 2], 'o-', linewidth=1, markersize=2, color=colors[n], label=f"Satellite {n+1}")
    ax.legend()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # 지구 그리기
    earth_radius = 6371000
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.3)

    plt.show()

def main():
    initial_positions = [
        np.array([7000.0, 0.0, 0.0])*1000,
        np.array([0.0, 7000.0, 0.0])*1000,
        np.array([0.0, 0.0, 7000.0])*1000,
        np.array([6800.0, 7000.0, 6500.0])*1000
    ]

    initial_velocities = [
        np.array([0.0, 7.5, 0.0])*1000,
        np.array([7.5, 0.0, 0.0])*1000,
        np.array([0.0, 0.0, 7.5])*1000,
        np.array([0.0, 7.5, 7.5])*1000
    ]

    satellites = [Satellite(initial_positions[i], initial_velocities[i]) for i in range(4)]
    timestep = 1
    steps = 3600
    mass = 1000
    orbits = simulate_satellite_orbits(satellites, timestep, steps, mass)
    plot_orbits(orbits)

main()
