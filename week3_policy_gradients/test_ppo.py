import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Định nghĩa lại bộ não PPO
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

print("\nĐang khởi động Agent PPO vô địch...")

# 2. Khởi tạo môi trường đồ họa
env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 3. Nạp "Linh hồn" vào
model = ActorCritic(state_dim, action_dim)
model.load_state_dict(torch.load("ppo_best.pth"))
model.eval() # Bật chế độ đi thi

# 4. Biểu diễn
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    with torch.no_grad():
        action_probs, _ = model(state_tensor)
        # Bỏ gieo xúc xắc, chọn luôn hành động tự tin nhất
        action = torch.argmax(action_probs).item()
        
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Hoàn thành xuất sắc! Tổng điểm: {total_reward}")
env.close()