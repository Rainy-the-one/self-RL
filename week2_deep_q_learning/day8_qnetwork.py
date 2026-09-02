import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

# 1. Khởi tạo môi trường CartPole
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0] # Input: 4 thông số (vị trí, vận tốc, góc, vận tốc góc)
action_dim = env.action_space.n            # Output: 2 hành động (Đẩy trái, Đẩy phải)

# 2. Xây dựng cấu trúc Mạng Nơ-ron (Q-Network)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Lớp ẩn thứ nhất (từ 4 input -> 64 nơ-ron)
        self.fc1 = nn.Linear(state_dim, 64)
        # Lớp ẩn thứ hai (từ 64 -> 64 nơ-ron)
        self.fc2 = nn.Linear(64, 64)
        # Lớp đầu ra (từ 64 -> 2 Q-values cho 2 action)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Hàm kích hoạt ReLU giúp mạng học được các logic phi tuyến tính
        x = F.relu(self.fc2(x))
        x = self.fc3(x)         # Output không dùng hàm kích hoạt vì Q-value có thể là số âm/dương bất kỳ
        return x

# 3. Khởi tạo bộ não
brain = QNetwork(state_dim, action_dim)
print(brain)

# 4. Thử nghiệm "chạy" một trạng thái qua não
state, _ = env.reset()
# PyTorch yêu cầu input phải là dạng Tensor (ma trận của Torch)
state_tensor = torch.FloatTensor(state) 

# Đưa state vào mạng nơ-ron để dự đoán Q-values
q_values = brain(state_tensor)

print("\nState hiện tại:", state)
print("Q-values dự đoán cho (Trái, Phải):", q_values.detach().numpy())