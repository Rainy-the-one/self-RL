import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 1. Define the Q-Network Architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. Define the Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)), 
                torch.LongTensor(actions), 
                torch.FloatTensor(rewards), 
                torch.FloatTensor(np.array(next_states)), 
                torch.FloatTensor(dones))
    
    def __len__(self):
        return len(self.buffer)

# 3. Initialize Environment and Hyperparameters
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

main_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(main_net.state_dict())  # Copy initial weights to target network

optimizer = optim.Adam(main_net.parameters(), lr=0.0005)
memory = ReplayBuffer()

batch_size = 128
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
target_update_freq = 10

print("Training DQN... (This will take about 1-2 minutes)")

# 4. Training Loop
for episode in range(500):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = main_net(torch.FloatTensor(state))
                action = torch.argmax(q_values).item()
        
        # Interact with the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Store transition in replay buffer
        memory.push(state, action, reward, next_state, done)
        state = next_state
        
        # Learn from experiences if enough samples are gathered
        if len(memory) > batch_size:
            states, actions, rewards, next_states, dones = memory.sample(batch_size)
            
            # Compute current Q values
            current_q = main_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q values using target network
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                target_q = rewards + gamma * max_next_q * (1 - dones)
            
            # Compute loss and update weights via gradient descent
            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Update target network weights periodically
    if episode % target_update_freq == 0:
        target_net.load_state_dict(main_net.state_dict())
        
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

# 5. Test the Trained Agent with Visual Rendering
print("\nStarting test simulation...")
env_test = gym.make("CartPole-v1", render_mode="human")
state, _ = env_test.reset()
done = False

while not done:
    with torch.no_grad():
        q_values = main_net(torch.FloatTensor(state))
        action = torch.argmax(q_values).item()
    state, reward, terminated, truncated, _ = env_test.step(action)
    done = terminated or truncated

env_test.close()