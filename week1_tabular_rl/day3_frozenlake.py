import gymnasium as gym
import time

# Initialize environment with human render mode to visualize actions
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

# Reset environment to initial state
state, info = env.reset()
print(f"Initial State: {state}")

# Run random simulation for 20 steps
for step in range(20):
    print(f"\n--- Step {step + 1} ---")
    
    # Choose a random action (0: Left, 1: Down, 2: Right, 3: Up)
    action = env.action_space.sample()
    print(f"Selected Action: {action}")
    
    # Send action to the environment
    next_state, reward, terminated, truncated, info = env.step(action)
    
    print(f"New State: {next_state}")
    print(f"Reward Received: {reward}")
    
    # Check if game is over
    if terminated or truncated:
        print("Game Over! Resetting environment...")
        state, info = env.reset()
    else:
        state = next_state  # Update state for the next step
        
    time.sleep(1)  # Pause for 1 second to observe visual updates

env.close()