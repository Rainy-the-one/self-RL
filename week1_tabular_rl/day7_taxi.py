import gymnasium as gym
import numpy as np
import random

# Initialize the environment
env = gym.make("Taxi-v4")

# Initialize Q-table with zeros
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# Hyperparameters
total_episodes = 5000
learning_rate = 0.8
gamma = 0.95
epsilon = 1.0
decay_rate = 0.005

# Training loop
for episode in range(total_episodes):
    state, info = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(q_table[state, :])  # Exploitation
        else:
            action = env.action_space.sample()     # Exploration
            
        # Take action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update Q-table using the Q-learning Bellman equation (Off-policy)
        q_table[state, action] += learning_rate * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        
        # Transition to the next state
        state = next_state
        
    # Decay epsilon after each episode
    epsilon = max(0.01, epsilon * np.exp(-decay_rate))

print("Training finished! Q-Table:\n")
print(np.round(q_table, 5))


# --- TESTING THE TRAINED AGENT ---
print("\nRunning test simulation with visual rendering...")
env_test = gym.make("Taxi-v4", render_mode="human")
state, info = env_test.reset()
done = False
total_reward = 0

while not done:
    # 100% exploitation during testing
    action = np.argmax(q_table[state, :])
    state, reward, terminated, truncated, info = env_test.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Trip completed! Total reward earned: {total_reward}")
env_test.close()