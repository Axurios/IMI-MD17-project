import numpy as np
import matplotlib.pyplot as plt
import verlet_muller

n = 5 #Nombre de particules
dt = 0.01 #Pas de temps
num_steps = 100  #Nombre d'itérations 
m = np.ones(n) #Masse des particules


#Initialisation des positions et des vitesses
q_0 = np.random.rand(n, 2)*2-1
p_0 = np.random.rand(n, 2)*2-1


trajectory, _ = verlet_muller.verlet_scheme_n_particule(q_0, p_0, dt, m ,num_steps)

#plot the trajectories
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)



potential_grid = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        potential_grid[i, j] = verlet_muller.grad_muller_potential(np.array(x[i], y[j]))

potential_grid[potential_grid > 1] = 1
potential_grid=np.transpose(potential_grid)



fig, ax = plt.subplots()


# Set the axis limits
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 2)

# Create a line plot for each particle
lines = [ax.plot([], [], 'o-', markersize=2)[0] for _ in range(n)]


contour= ax.contourf(x, y, potential_grid, levels=100, cmap='viridis')

cbar = plt.colorbar(contour, ax=ax, label='Muller Potential')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Muller Brown')



def init():
    for line in lines:
        line.set_data([], [])
    return lines


def update(frame):
    for i, line in enumerate(lines):
        line.set_data(trajectory[i, :frame, 0], trajectory[i, :frame, 1])
        line.set_color('black')#plt.cm.viridis(i/n))  # Use a colormap for different colors
    return lines

# Run the simulation

# Create the animation
animation = verlet_muller.FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True)

plt.show()
