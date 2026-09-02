import gymnasium as gym
import numpy as np
import random

# 1. Khởi tạo môi trường (Lưới 4x4, không trơn trượt)
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# 2. Khởi tạo Bảng Q-Table với toàn số 0
state_size = env.observation_space.n  # 16 trạng thái (từ ô 0 đến 15)
action_size = env.action_space.n      # 4 hành động (Trái, Xuống, Phải, Lên)
q_table = np.zeros((state_size, action_size))

# 3. Cài đặt Siêu tham số (Hyperparameters)
total_episodes = 10000         # Số lần Agent chơi lại từ đầu
learning_rate = 0.8           # Alpha: Tốc độ học (Cập nhật bao nhiêu % kiến thức mới)
gamma = 0.95                  # Hệ số chiết khấu: Tầm nhìn xa
epsilon = 0.1                 # Khởi đầu khám phá 100%
max_epsilon = 1.0             
min_epsilon = 0.01            # Giữ lại 1% khám phá ở cuối để đề phòng
decay_rate = 0.005            # Tốc độ giảm epsilon sau mỗi ván chơi

print("Đang huấn luyện Agent... Vui lòng đợi.\n")

# 4. BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN
for episode in range(total_episodes):
    state, info = env.reset()
    done = False
    
    while not done:
        # --- BƯỚC A: CHIẾN LƯỢC EPSILON-GREEDY ---
        # Tung đồng xu ngẫu nhiên từ 0 đến 1
        tradeoff = random.uniform(0, 1)
        
        if tradeoff > epsilon:
            # KHAI THÁC (Exploitation): Chọn hành động có điểm cao nhất trong Q-Table
            action = np.argmax(q_table[state, :])
        else:
            # KHÁM PHÁ (Exploration): Chọn bừa một hành động
            action = env.action_space.sample()
            
        # --- BƯỚC B: TƯƠNG TÁC VỚI MÔI TRƯỜNG ---
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- BƯỚC C: CẬP NHẬT Q-TABLE (PHƯƠNG TRÌNH BELLMAN) ---
        # Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        
        # Chuyển sang trạng thái mới
        state = next_state
        
    # --- BƯỚC D: GIẢM EPSILON ---
    # Sau mỗi ván, giảm tỷ lệ khám phá xuống một chút (Epsilon Decay)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print("Huấn luyện hoàn tất! Đây là bộ não (Q-Table) của Agent:\n")
print(np.round(q_table, 3))

# 5. XEM KẾT QUẢ: CHO AGENT CHƠI THỰC TẾ VỚI BỘ NÃO ĐÃ HỌC
print("\n--- KIỂM TRA AGENT ---")
env_test = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")
state, info = env_test.reset()
done = False

while not done:
    # Lúc này không khám phá nữa, chỉ dùng 100% Khai thác
    action = np.argmax(q_table[state, :])
    state, reward, terminated, truncated, info = env_test.step(action)
    done = terminated or truncated

print("Tuyệt vời! Agent đã lấy được kho báu!" if reward == 1 else "Agent vẫn bị rớt hố :(")
env_test.close()