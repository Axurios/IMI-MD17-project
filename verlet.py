import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



K = np.array([-20, -10, -17, 1.5])
a = np.array([-1, -1, -6.5, 0.7])
b = np.array([0, 0, 11, 15])
c = np.array([-10, -10, -6.5, 0.7])
beta = np.array([1, 0, -0.5, -1] )
gamma = np.array([0, 0.5, 1.5, 1])

# potentiel de Muller
def potential(x, y):
    S=0
    for i in range (4):
        S+= K[i]*np.exp(a[i] * (x - beta[i])**2 + b[i] * (x - beta[i])*(y- gamma[i])+ c[i]*(y - gamma[i])**2)
    return S

# calcul des dérivées partielles
def gradient_x(x, y):
    S=0
    for i in range (4):
        S+= K[i]* (2* a[i] * (x - beta[i]) + b[i] *(y- gamma[i]))*np.exp(a[i] * (x - beta[i])**2 + b[i] * (x - beta[i])*(y- gamma[i])+ c[i]*(y - gamma[i])**2)
    return S

# Gradient du potentiel de Muller par rapport à y
def gradient_y(x, y):
    S=0
    for i in range (0, 4):
        S+= K[i]* (2* c[i]*(y - gamma[i]) + b[i] *(x - beta[i]))*np.exp(a[i] * (x - beta[i])**2 + b[i] * (x - beta[i])*(y- gamma[i])+ c[i]*(y - gamma[i])**2)
    return S




def verlet_single_particle(q, p, dt, m, num_steps, grad_cache):
    trajectory = np.zeros((num_steps, 2))
    momenta = np.zeros((num_steps, 2))
    trajectory[0] = q  
    momenta[0] = p

    for i in range(1, num_steps):
        
        if grad_cache is None:
            grad_cache = np.array([gradient_x(trajectory[i-1][0], trajectory[i-1][1]), gradient_y(trajectory[i-1][0], trajectory[i-1][1])])
        p_half = momenta[i-1] - 0.5 * dt * grad_cache

        M = m * np.eye(2)

        q_new = trajectory[i-1] + dt * np.linalg.solve(M, p_half)

        grad_cache_new = np.array([gradient_x(q_new[0], q_new[1]), gradient_y(q_new[0], q_new[1])])
        grad_cache = grad_cache_new

        p_new = p_half - 0.5 * dt * grad_cache_new

        trajectory[i] = q_new
        momenta[i] = p_new

    return trajectory, momenta


def verlet(N, q0, p0, dt, num_steps, m):
    grad_caches = [None] * N

    trajectory = np.zeros((N, num_steps, 2))
    momenta = np.zeros((N, num_steps, 2))

    for i in range(N):
        grad_i = grad_caches[i]
        trajectory[i], momenta[i] = verlet_single_particle(q0[i], p0[i], dt, m[i], num_steps, grad_i)

    return trajectory, momenta






# parameters
N = 10  
dt = 0.01 
num_steps = 500  
m = [1.0]*N  

# Initialize q0 and p0 
q0 = np.array([[0,0]]*N)
p0 = np.array([[0,0]]*N)
for i in range(N):
    q0[i][0]= np.random.rand()
    q0[i][1]= np.random.rand()
    p0[i][0]= np.random.rand()
    p0[i][1]= np.random.rand()


fig, ax = plt.subplots()

# Set the axis limits
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Create a line plot for each particle
lines = [ax.plot([], [], 'o-', markersize=2)[0] for _ in range(N)]

def init():
    for line in lines:
        line.set_data([], [])
    return lines


def update(frame):
    for i, line in enumerate(lines):
        line.set_data(trajectory[i, :frame, 0], trajectory[i, :frame, 1])
        line.set_color(plt.cm.viridis(i/N))  # Use a colormap for different colors
    return lines

# Run the simulation
trajectory, _ = verlet(N, q0, p0, dt, num_steps, m)

# Create the animation
animation = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True)

# Show the animation
plt.show()
