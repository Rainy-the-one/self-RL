import gymnasium as gym
import numpy as np
import random

# Initialize the environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
q_table = np.zeros((16, 4))

# Hyperparameters
total_episodes = 10000
learning_rate = 0.8
gamma = 0.95
epsilon = 1.0
decay_rate = 0.005

# Training loop
for episode in range(total_episodes):
    state, info = env.reset()
    done = False
    
    # Choose the first action of the episode (Epsilon-greedy)
    if random.uniform(0, 1) > epsilon:
        action = np.argmax(q_table[state, :])
    else:
        action = env.action_space.sample()
        
    while not done:
        # Interact with the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Choose the next action (On-policy characteristic of SARSA)
        if not done:
            if random.uniform(0, 1) > epsilon:
                next_action = np.argmax(q_table[next_state, :])
            else:
                next_action = env.action_space.sample()
        else:
            next_action = 0
            
        # Update Q-table using the SARSA update rule
        q_table[state, action] += learning_rate * (
            reward + gamma * q_table[next_state, next_action] - q_table[state, action]
        )
        
        # Move to the next step
        state = next_state
        action = next_action
        
    # Decay epsilon
    epsilon = max(0.01, epsilon * np.exp(-decay_rate))

print("SARSA Q-Table:\n", np.round(q_table, 3))


# --- TESTING THE SARSA AGENT ---
print("\n--- TESTING SARSA AGENT ---")
env_test = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")
state, info = env_test.reset()
done = False

while not done:
    # Use 100% exploitation during testing
    action = np.argmax(q_table[state, :])
    state, reward, terminated, truncated, info = env_test.step(action)
    done = terminated or truncated

print("Completed!", "Reached the goal 🏆" if reward == 1 else "Fell into a hole 💀")
env_test.close()