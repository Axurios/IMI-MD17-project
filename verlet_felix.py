import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


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


trajectory, _ = verlet_scheme_n_particule(n, q_0, p_0, dt, m ,num_steps)


x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

z=np.meshgrid(x,y)


potential_grid = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        potential_grid[i, j] = muller_potential(x[i], y[j])

#crop all value above 1 to 1
potential_grid[potential_grid > 1] = 1


plt.figure(figsize=(8, 6))
plt.contourf(x, y, potential_grid, levels=50, cmap='viridis')
plt.colorbar(label='Muller Potential')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Muller Potential Heat Map')
plt.show()


print(muller_potential(-2, 2))