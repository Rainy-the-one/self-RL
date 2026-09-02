import gymnasium as gym
import numpy as np
import random

# Khởi tạo môi trường
env = gym.make("Taxi-v4")

# ==========================================
# TODO 1: Khởi tạo ma trận Q-Table toàn số 0
# Gợi ý: Dùng env.observation_space.n (số hàng) và env.action_space.n (số cột)
# ==========================================
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size,action_size))

# Siêu tham số
total_episodes = 5000
learning_rate = 0.8
gamma = 0.95
epsilon = 1.0
decay_rate = 0.005

for episode in range(total_episodes):
    state, info = env.reset()
    done = False
    
    while not done:
        # ==========================================
        # TODO 2: Cài đặt chiến lược Epsilon-Greedy
        # Gợi ý: Tung đồng xu ngẫu nhiên từ 0 đến 1. 
        # Nếu lớn hơn epsilon -> Khai thác (dùng np.argmax)
        # Nếu nhỏ hơn epsilon -> Khám phá (dùng env.action_space.sample())
        # ==========================================
        
        # ---> VIẾT CODE CỦA BẠN Ở ĐÂY <---
        tradeoff = random.uniform(0, 1)

        if tradeoff > epsilon:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        
        # Taxi thực hiện hành động
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # ==========================================
        # TODO 3: Cập nhật Q-Table bằng phương trình Q-Learning (Off-policy)
        # Gợi ý công thức: Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s', a')) - Q(s,a)]
        # ==========================================
        
        # ---> VIẾT CODE CỦA BẠN Ở ĐÂY <---
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
        
        # Chuẩn bị cho bước lặp tiếp theo
        state = next_state
        
    # Giảm dần epsilon
    epsilon = max(0.01, epsilon * np.exp(-decay_rate))

print("Huấn luyện xong! Q-Table đã sẵn sàng.")
print(np.round(q_table, 5))


# --- KIỂM TRA TAXI TRONG THỰC TẾ ---
print("\nBắt đầu chạy thử nghiệm với render đồ họa...")
env_test = gym.make("Taxi-v4", render_mode="human")
state, info = env_test.reset()
done = False
total_reward = 0

while not done:
    # Agent sử dụng 100% kinh nghiệm (Exploitation)
    action = np.argmax(q_table[state, :])
    state, reward, terminated, truncated, info = env_test.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Hoàn thành chuyến đi! Tổng điểm (Reward) thu được: {total_reward}")
env_test.close()