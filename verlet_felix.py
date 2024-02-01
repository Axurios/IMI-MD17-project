import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



K = np.array([-20, -10, -17, 1.5])
a = np.array([-1, -1, -6.5, 0.7])
b = np.array([0, 0, 11, 15])
c = np.array([-10, -10, -6.5, 0.7])
beta = np.array([1, 0, -0.5, -1] )
gamma = np.array([0, 0.5, 1.5, 1])

def muller_potential(x, y):
    return np.dot(K, np.exp(a * (x - beta)**2 + b * (x - beta)*(y- gamma)+ c*(y - gamma)**2))

def grad_muller_potential_x(x, y):
    return np.dot(K, (2* a * (x - beta) + b *(y- gamma))*np.exp(a * (x - beta)**2 + b * (x - beta)*(y- gamma)+ c*(y - gamma)**2))

def grad_muller_potential_y(x, y):
    return np.dot(K, (2* c*(y - gamma) + b *(x - beta))*np.exp(a * (x - beta)**2 + b * (x - beta)*(y- gamma)+ c*(y - gamma)**2))

def grad_muller_potential(q):
    return np.array([grad_muller_potential_x(q[0], q[1]), 
                     grad_muller_potential_y(q[0], q[1])])


def verlet_scheme_one_particule(q_0, p_0, dt, m, num_steps):
    q = np.zeros((num_steps, 2))
    p = np.zeros((num_steps, 2))

    q[0] = q_0 
    p[0] = p_0

    grad_v= grad_muller_potential(q_0)

    for i in range(1, num_steps):
        p[i]=p[i-1] - dt*grad_v/2
        q[i]=q[i-1] + dt*p[i]/m
        grad_v= grad_muller_potential(q[i])
        p[i]=p[i] - dt*grad_v/2

    return q, p


def verlet_scheme_n_particule(n, q_0, p_0, dt, m, num_steps):

    q = np.zeros((n, num_steps, 2))
    p = np.zeros((n, num_steps, 2))

    for i in range(n):
        q[i], p[i] = verlet_scheme_one_particule(q_0[i], p_0[i], 
                                                 dt, m[i], num_steps)

    return q, p



n = 10  
dt = 0.01 
num_steps = 500  
m = np.ones(n)

q_0 = np.random.rand(n, 2)
p_0 = np.random.rand(n, 2)



fig, ax = plt.subplots()

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

lines = [ax.plot([], [], 'o-', markersize=2)[0] for _ in range(n)]

def init():
    for line in lines:
        line.set_data([], [])
    return lines    

trajectory, _ = verlet_scheme_n_particule(n, q_0, p_0, dt, m ,num_steps)

def update(frame ):
    for i, line in enumerate(lines):
        line.set_data(trajectory[i, :frame, 0], trajectory[i, :frame, 1])
        line.set_color(plt.cm.viridis(i/n))  # Use a colormap for different colors
    return lines



# Create the animation
animation = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True)

# Show the animation
plt.show()
