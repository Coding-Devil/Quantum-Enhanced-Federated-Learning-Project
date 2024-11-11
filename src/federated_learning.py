# src/federated_learning.py

import syft as sy
import torch
import torch.nn as nn
import torch.optim as optim

# Hook PyTorch
hook = sy.TorchHook(torch)

# Create virtual workers
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# Sample model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # Adjust input size as needed
        self.fc2 = nn.Linear(5, 1)    # Adjust output size as needed

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, data, target, worker):
    """Train the model on a specific worker."""
    model.send(worker)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Send data to the worker
    data = data.send(worker)
    target = target.send(worker)

    # Training loop
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Get model back
    model.get()

if __name__ == "__main__":
    # Instantiate model
    model = SimpleNN()

    # Example training loop
    for epoch in range(10):  # Number of epochs
        # Load your data for each worker
        # data_alice, target_alice = ...
        # data_bob, target_bob = ...
        
        # Train on Alice's data
        train_model(model, data_alice, target_alice, alice)
        
        # Train on Bob's data
        train_model(model, data_bob, target_bob, bob)