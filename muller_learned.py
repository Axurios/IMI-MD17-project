import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import time 
# Load training data from CSV file
training_data = pd.read_csv('training_data.csv')

# Normalize x and y
#training_data['x'] = (training_data['x'] - training_data['x'].mean()) / training_data['x'].std()
#training_data['y'] = (training_data['y'] - training_data['y'].mean()) / training_data['y'].std()

# Convert DataFrame to PyTorch tensors
x_data = torch.tensor(training_data[['x', 'y']].values, dtype=torch.float32)
grad_data = torch.tensor(training_data[['grad_x', 'grad_y']].values, dtype=torch.float32)

# Split the data into training and validation sets
x_train, x_val, grad_train, grad_val = train_test_split(x_data, grad_data, test_size=0.2, random_state=42)


#print min grad


cliped_value=1000

#clip grad at cliped_value
grad_train[grad_train > cliped_value] = cliped_value
grad_val[grad_val > cliped_value] = cliped_value

#clip grad at -cliped_value
grad_train[grad_train < -cliped_value] = -cliped_value
grad_val[grad_val < -cliped_value] = -cliped_value

# Define the neural network model
class GradientModel(nn.Module):
    def __init__(self):
        super(GradientModel, self).__init__()
        #Four layers
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
    
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class PotentialGradientModel(nn.Module):
    def __init__(self):
        super(PotentialGradientModel, self).__init__()
        # Four layers for potential
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

        # Four layers for gradient
        self.fc5 = nn.Linear(2, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc8 = nn.Linear(128, 2)

    def forward(self, x):
        # Potential
        x_potential = torch.tanh(self.fc1(x))
        x_potential = torch.tanh(self.fc2(x_potential))
        x_potential = torch.tanh(self.fc3(x_potential))
        potential = self.fc4(x_potential)

        # Gradient
        x_gradient = torch.tanh(self.fc5(x))
        x_gradient = torch.tanh(self.fc6(x_gradient))
        x_gradient = torch.tanh(self.fc7(x_gradient))
        gradient = self.fc8(x_gradient)

        return potential, gradient



"""
criterion_potential = nn.MSELoss()
criterion_gradient = nn.MSELoss()

for inputs, grads in train_dataloader:
    optimizer.zero_grad()
    potential_outputs, gradient_outputs = model(inputs)

    # Compute losses
    potential_loss = criterion_potential(potential_outputs, ground_truth_potential)
    gradient_loss = criterion_gradient(gradient_outputs, ground_truth_gradient)

    # Total loss
    total_loss = potential_loss + gradient_loss

    # Backward and optimize
    total_loss.backward()
    optimizer.step()
"""


"""
# Instantiate the model, loss function, and optimizer
model = GradientModel()
#model.apply(weights_init)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Create DataLoader for training and validation sets
train_dataset = TensorDataset(x_train, grad_train)
val_dataset = TensorDataset(x_val, grad_val)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Train the model with validation
epochs = 1000
for epoch in range(epochs):
    model.train()
    for inputs, grads in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, grads)


        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, grads in val_dataloader:
            outputs = model(inputs)
            val_loss += criterion(outputs, grads).item()

    val_loss /= len(val_dataloader)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')



torch.save(model.state_dict(), 'trained_model.pth')
"""