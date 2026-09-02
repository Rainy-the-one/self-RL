import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# 1. Define the Policy Network Architecture (Softmax Output)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def select_action(policy_net, state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    probs = policy_net(state_tensor)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def update_policy(optimizer, log_probs, rewards, gamma=0.99):
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    
    discounted_rewards = torch.FloatTensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    
    policy_loss = []
    for log_prob, G_t in zip(log_probs, discounted_rewards):
        policy_loss.append(-log_prob * G_t)
        
    optimizer.zero_grad()
    sum(policy_loss).backward()
    optimizer.step()

# 2. Initialize Environment and Model
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

print("Training REINFORCE...")

# 3. Training Loop (Monte Carlo evaluation at the end of each episode)
for episode in range(500):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    done = False
    total_reward = 0
    
    while not done:
        action, log_prob = select_action(policy_net, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
        state = next_state
        
    # Update network weights AFTER the episode terminates
    update_policy(optimizer, log_probs, rewards)
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# 4. Test the Trained Agent
print("\nStarting test simulation...")
env_test = gym.make("CartPole-v1", render_mode="human")
state, _ = env_test.reset()
done = False
while not done:
    action, _ = select_action(policy_net, state)
    state, reward, terminated, truncated, _ = env_test.step(action)
    done = terminated or truncated
env_test.close()