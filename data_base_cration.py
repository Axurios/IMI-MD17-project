import verlet_muller
import numpy as np

num_samples = 100000  # Nombre de points de données
q_train = np.random.uniform(low=-2, high=4, size=(num_samples,2))

# Calculer le potentiel associé à chaque paire (x, y)
potential_values=[]
for i in range(num_samples):
    potential_values.append(verlet_muller.grad_muller_potential(q_train[i]))


# Créer la base de données d'entraînement x,y,potential_x,potential_y
training_data = np.hstack((q_train, np.array(potential_values)))

# Enregistrez la base de données dans un fichier CSV, par exemple
np.savetxt('training_data.csv', training_data, delimiter=',', header='x,y,grad_x,grad_y', comments='')

# Affichez quelques exemples
print("Exemples de données générées:")
print(training_data[:5])