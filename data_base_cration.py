import verlet_felix
import numpy as np

num_samples = 10000  # Nombre de points de données
q_train = np.random.uniform(low=-2, high=4, size=(num_samples,2))

# Calculer le potentiel associé à chaque paire (x, y)
potential_values = verlet_felix.grad_muller_potential(q_train)

# Créer la base de données d'entraînement
training_data = np.column_stack((q_train, potential_values))

# Enregistrez la base de données dans un fichier CSV, par exemple
np.savetxt('training_data.csv', training_data, delimiter=',', header='x,y,potential', comments='')

# Affichez quelques exemples
print("Exemples de données générées:")
print(training_data[:5])