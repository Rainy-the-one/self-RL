import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.actor(x), dim=1)
        state_value = self.critic(x)
        return action_probs, state_value

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCriticNetwork(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.002)
gamma = 0.99
best_reward = 0

# ==============================================================================
# TOGGLE BETWEEN THE ORIGINAL AND STABILIZED VERSIONS:
# ==============================================================================
USE_STABILIZED_VERSION = True  # True: Stabilized version | False: Original version

if not USE_STABILIZED_VERSION:
    print(">>> Running: ORIGINAL Actor-Critic (Prone to high variance and policy collapse)...")
else:
    print(">>> Running: STABILIZED Actor-Critic (With Loss Balancing, Entropy & Gradient Clipping)...")


for episode in range(1000):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = model(state_tensor)
        
        m = Categorical(action_probs)
        action = m.sample()
        
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward
        
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        _, next_state_value = model(next_state_tensor)
        
        td_target = reward + gamma * next_state_value * (1 - int(done))
        advantage = td_target - state_value
        
        # ----------------------------------------------------------------------
        # BLOCK 1: ORIGINAL VERSION
        # Lacks reward shaping, loss balancing, entropy, and clipping. 
        # Often suffers from high variance and sudden training instability.
        # ----------------------------------------------------------------------
        if not USE_STABILIZED_VERSION:
            actor_loss = -m.log_prob(action) * advantage.detach()
            critic_loss = F.mse_loss(state_value, td_target.detach())
            total_loss = actor_loss + critic_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        # ----------------------------------------------------------------------
        # BLOCK 2: STABILIZED VERSION (3 SHIELDS)
        # - Negative reward shaping (-10) upon failing early to learn failure boundaries.
        # - Balanced loss weights (0.5 for critic) + Entropy bonus for exploration.
        # - Gradient Clipping (max_norm=1.0) to prevent exploding gradients.
        # ----------------------------------------------------------------------
        else:
            if done and total_reward < 500:
                reward = -10  # Negative penalty for premature failure

            critic_loss = F.mse_loss(state_value, td_target.detach())
            actor_loss = -m.log_prob(action) * advantage.detach()
            entropy = m.entropy()  # Shield 2: Encourages exploration
            
            # Shield 1: Weighted loss combination
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Shield 3: Gradient Clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        state = next_state
        
    if total_reward > best_reward:
        best_reward = total_reward
        print(f"New record at episode {episode}: {best_reward} points! Model saved.")
        torch.save(model.state_dict(), "actor_critic_best.pth")

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

print("\nTraining complete!")