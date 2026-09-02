import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

# 1. Initialize the CartPole environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]  # Input: 4 state features (position, velocity, angle, angular velocity)
action_dim = env.action_space.n             # Output: 2 discrete actions (Push left, Push right)

# 2. Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # First hidden layer (4 inputs -> 64 neurons)
        self.fc1 = nn.Linear(state_dim, 64)
        # Second hidden layer (64 -> 64 neurons)
        self.fc2 = nn.Linear(64, 64)
        # Output layer (64 -> 2 Q-values for 2 actions)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU activation for non-linearity
        x = F.relu(self.fc2(x))
        x = self.fc3(x)          # Raw outputs since Q-values can be any real number
        return x

# 3. Initialize the neural network model
brain = QNetwork(state_dim, action_dim)
print(brain)

# 4. Test a forward pass with an initial state
state, _ = env.reset()
state_tensor = torch.FloatTensor(state)  # Convert state numpy array to PyTorch Tensor

# Pass the tensor through the network to get predicted Q-values
q_values = brain(state_tensor)

print("\nCurrent State:", state)
print("Predicted Q-values for actions (Left, Right):", q_values.detach().numpy())
env.close()