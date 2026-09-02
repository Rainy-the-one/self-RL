import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage3_policy_gradients.day12_actor_critic import ActorCriticNetwork

# Nếu chạy file mới, bạn cần định nghĩa lại class ActorCriticNetwork ở đây...

print("\nĐang nạp file trọng số 'actor_critic_best.pth'...")

# 1. Khởi tạo môi trường đồ họa
env_test = gym.make("CartPole-v1", render_mode="human")
state_dim = env_test.observation_space.shape[0]
action_dim = env_test.action_space.n

# 2. Tạo hình hài bộ não và nạp "linh hồn" vào
best_model = ActorCriticNetwork(state_dim, action_dim)
best_model.load_state_dict(torch.load("actor_critic_best.pth"))

# Chuyển mạng sang chế độ đánh giá (bắt buộc trong PyTorch khi test)
best_model.eval() 

# 3. Chạy vòng lặp biểu diễn
state, _ = env_test.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    # Tắt tính toán đạo hàm để tiết kiệm RAM và tăng tốc CPU
    with torch.no_grad():
        action_probs, _ = best_model(state_tensor)
        # Chuyển từ "Gieo xúc xắc" sang "Lấy mốc cao nhất"
        action = torch.argmax(action_probs).item()
        
    state, reward, terminated, truncated, _ = env_test.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Màn biểu diễn kết thúc! Điểm số: {total_reward}")
env_test.close()