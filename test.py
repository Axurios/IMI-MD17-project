import verlet
import verlet_felix
import numpy as np
import matplotlib.pyplot as plt
import heatmap_verlet


N = 1  
dt = 0.01 
num_steps = 1000
m = np.ones(N)

q_0 = np.random.rand(N, 2)
p_0 = np.random.rand(N, 2)
trajectoires_1,_=heatmap_verlet.verlet(N, q_0, p_0, dt, num_steps, m)
trajectoires_2,_=verlet_felix.verlet_scheme_n_particule(q_0, p_0, dt, m, num_steps)

#heat map
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

potential_grid_1 = np.zeros((len(x), len(y)))
potential_grid_2 = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        potential_grid_1[i, j] = verlet_felix.muller_potential(x[i], y[j])
        potential_grid_2[i, j] = heatmap_verlet.potential(x[i], y[j])
potential_grid_1[potential_grid_1 > 1] = 1
potential_grid_2[potential_grid_2 > 1] = 1
#transpose potential_grid_2 
potential_grid_1=np.transpose(potential_grid_1)
potential_grid_2=np.transpose(potential_grid_2)


#plot heat map and trajectories on same graph

fig, ax = plt.subplots()
ax.imshow(potential_grid_1, extent=[-5, 5, -5, 5])
ax.imshow(potential_grid_2, extent=[-5, 5, -5, 5])
ax.plot(trajectoires_1[0,:,0],trajectoires_1[0,:,1],label="verlet")
ax.plot(trajectoires_2[0,:,0],trajectoires_2[0,:,1],label="verlet_felix")
ax.legend()
plt.show()

"""
plt.plot(trajectoires_1[0,:,0],trajectoires_1[0,:,1],label="verlet")
plt.legend()
plt.show()

plt.plot(trajectoires_2[0,:,0],trajectoires_2[0,:,1],label="verlet_felix")
plt.legend()
plt.show()

"""