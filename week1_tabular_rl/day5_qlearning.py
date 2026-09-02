import gymnasium as gym
import numpy as np
import random

# Initialize the environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# Initialize Q-table
state_size = env.observation_space.n  # 16 states (0 to 15)
action_size = env.action_space.n      # 4 actions (Left, Down, Right, Up)
q_table = np.zeros((state_size, action_size))

# Hyperparameters
total_episodes = 10000
learning_rate = 0.8
gamma = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

print("Training agent... Please wait.\n")

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
            
        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update Q-table using the Bellman equation
        q_table[state, action] += learning_rate * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        
        # Move to the next state
        state = next_state
        
    # Exponential decay for epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print("Training complete! Learned Q-Table:\n")
print(np.round(q_table, 3))


# --- TESTING THE AGENT ---
print("\n--- TESTING AGENT ---")
env_test = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")
state, info = env_test.reset()
done = False

while not done:
    # Pure exploitation during test mode
    action = np.argmax(q_table[state, :])
    state, reward, terminated, truncated, info = env_test.step(action)
    done = terminated or truncated

print("Awesome! The agent reached the treasure!" if reward == 1 else "The agent fell into a hole :(")
env_test.close()