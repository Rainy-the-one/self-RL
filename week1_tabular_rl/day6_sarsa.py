import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
q_table = np.zeros((16, 4))

total_episodes = 10000
learning_rate = 0.8
gamma = 0.95
epsilon = 1.0
decay_rate = 0.005

for episode in range(total_episodes):
    state, info = env.reset()
    done = False
    
    # BƯỚC A: Chọn hành động ĐẦU TIÊN của ván game
    if random.uniform(0, 1) > epsilon:
        action = np.argmax(q_table[state, :])
    else:
        action = env.action_space.sample()
        
    while not done:
        # BƯỚC B1: Tương tác với môi trường
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # BƯỚC B2: Chọn LUÔN hành động tiếp theo (Khác biệt của SARSA)
        if not done:
            if random.uniform(0, 1) > epsilon:
                next_action = np.argmax(q_table[next_state, :])
            else:
                next_action = env.action_space.sample()
        else:
            next_action = 0 # Nếu game over thì không cần quan tâm next_action
            
        # BƯỚC C: Cập nhật Q-Table bằng hành động THỰC TẾ (next_action)
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + gamma * q_table[next_state, next_action] - q_table[state, action]
        )
        
        # Tiến lên bước tiếp theo
        state = next_state
        action = next_action
        
    epsilon = max(0.01, epsilon * np.exp(-decay_rate))

print("Q-Table của SARSA:\n", np.round(q_table, 3))
# 5. XEM KẾT QUẢ: CHO AGENT CHƠI THỰC TẾ
print("\n--- KIỂM TRA AGENT SARSA ---")
# Mở render_mode="human" để xem Agent chạy
env_test = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")
state, info = env_test.reset()
done = False

while not done:
    # Ở bước Test, Agent chỉ dùng 100% Khai thác (Exploitation) từ Q-Table
    action = np.argmax(q_table[state, :]) 
    state, reward, terminated, truncated, info = env_test.step(action)
    done = terminated or truncated

print("Hoàn thành! Kết quả:", "Lấy được kho báu 🏆" if reward == 1 else "Vẫn rớt hố 💀")
env_test.close()