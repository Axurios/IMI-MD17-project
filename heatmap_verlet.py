import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


K = np.array([-20, -10, -17, 1.5])
a = np.array([-1, -1, -6.5, 0.7])
b = np.array([0, 0, 11, 15])
c = np.array([-10, -10, -6.5, 0.7])
beta = np.array([1, 0, -0.5, -1])
gamma = np.array([0, 0.5, 1.5, 1])


# potentiel de Muller
def potential(x, y):
    S = 0
    for i in range(4):
        S += K[i] * np.exp(
            a[i] * (x - beta[i]) ** 2
            + b[i] * (x - beta[i]) * (y - gamma[i])
            + c[i] * (y - gamma[i]) ** 2
        )  # noqa:
    return S


# calcul des dérivées partielles
def gradient_x(x, y):
    S = 0
    for i in range(4):
        S += (
            K[i]
            * (2 * a[i] * (x - beta[i]) + b[i] * (y - gamma[i]))
            * np.exp(
                a[i] * (x - beta[i]) ** 2
                + b[i] * (x - beta[i]) * (y - gamma[i])
                + c[i] * (y - gamma[i]) ** 2
            )
        )
    return S


# Gradient du potentiel de Muller par rapport à y
def gradient_y(x, y):
    S = 0
    for i in range(0, 4):
        S += (
            K[i]
            * (2 * c[i] * (y - gamma[i]) + b[i] * (x - beta[i]))
            * np.exp(
                a[i] * (x - beta[i]) ** 2
                + b[i] * (x - beta[i]) * (y - gamma[i])
                + c[i] * (y - gamma[i]) ** 2
            )
        )
    return S


def verlet_single_particle(q, p, dt, m, num_steps, grad_cache):
    trajectory = np.zeros((num_steps, 2))
    momenta = np.zeros((num_steps, 2))
    trajectory[0] = q
    momenta[0] = p

    for i in range(1, num_steps):
        if grad_cache is None:
            grad_cache = np.array(
                [
                    gradient_x(trajectory[i - 1][0], trajectory[i - 1][1]),
                    gradient_y(trajectory[i - 1][0], trajectory[i - 1][1]),
                ]
            )
        p_half = momenta[i - 1] - 0.5 * dt * grad_cache

        M = m * np.eye(2)

        q_new = trajectory[i - 1] + dt * np.linalg.solve(M, p_half)

        grad_cache_new = np.array(
            [gradient_x(q_new[0], q_new[1]), gradient_y(q_new[0], q_new[1])]
        )
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
        trajectory[i], momenta[i] = verlet_single_particle(
            q0[i], p0[i], dt, m[i], num_steps, grad_i
        )

    return trajectory, momenta


# Définition des paramètres
N = 10
dt = 0.01
num_steps = 500
m = [1.0] * N

# Initialize q0 and p0
q0 = np.array([[0, 0]] * N)
p0 = np.array([[0, 0]] * N)
for i in range(N):
    q0[i][0] = np.random.rand()
    q0[i][1] = np.random.rand()
    p0[i][0] = np.random.rand()
    p0[i][1] = np.random.rand()

trajectory, _ = verlet(N, q0, p0, dt, num_steps, m)

"""
# initialize the plot
def init():
    part_fig.set_data([], [])
    history_fig.set_data([], [])
    return part_fig, history_fig

# update the plot for each frame
def update(frame, line_main, line_history):
    x = trajectory[0, :frame, 0]  # Assuming you have only one line for the particle
    y = trajectory[0, :frame, 1]

    hist_x = []
    hist_y = []
    line_main.set_data(x, y)
    hist_x.append(x)
    hist_y.append(y)
    line_history.set_data(hist_x, hist_y)

    return line_main,


x_range = np.arange(-0.35, 1.9, 0.001)
y_range = np.arange(-0.3, 0.8, 0.001)
X, Y = np.meshgrid(x_range, y_range)
Z = potential(X,Y)

fig = plt.figure()
sep_bound = 30
contour_lines = np.arange(-250.0, 400, sep_bound)
back_fig = plt.contourf(X, Y, Z, 50, alpha=.75, cmap='rainbow', extend="both", levels=contour_lines)
part_fig, = plt.plot(q0[0], q0[1], ls='None', lw=1.0, color='blue', marker='o', ms=8, alpha=1)
history_fig, = plt.plot([], [], lw=2.0, color='#2c70a3', alpha=0.5)

# animation
animation = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, fargs=(part_fig, history_fig))
plt.show()

"""
