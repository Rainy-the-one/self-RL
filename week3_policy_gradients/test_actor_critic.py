import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the network architecture from the training script
from week3_policy_gradients.day12_actor_critic import ActorCriticNetwork

print("\nLoading weights from 'actor_critic_best.pth'...")

# 1. Initialize environment with visual rendering
env_test = gym.make("CartPole-v1", render_mode="human")
state_dim = env_test.observation_space.shape[0]
action_dim = env_test.action_space.n

# 2. Instantiate the model and load trained weights
best_model = ActorCriticNetwork(state_dim, action_dim)
best_model.load_state_dict(torch.load("actor_critic_best.pth"))

# Set model to evaluation mode (mandatory in PyTorch during testing)
best_model.eval() 

# 3. Run visual performance loop
state, _ = env_test.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    # Disable gradient computation to save memory and speed up CPU execution
    with torch.no_grad():
        action_probs, _ = best_model(state_tensor)
        # Select the deterministic greedy action (highest probability)
        action = torch.argmax(action_probs).item()
        
    state, reward, terminated, truncated, _ = env_test.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Performance finished! Total Score: {total_reward}")
env_test.close()