import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


# 1. Kiến trúc Não bộ (Giữ nguyên như Actor-Critic)
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

# 2. Siêu tham số PPO
update_timestep = 2000   # Gom đủ 2000 bước đi mới học 1 lần (Batching)
K_epochs = 4             # Đem rổ dữ liệu ra học đi học lại 4 lần
eps_clip = 0.2           # Kẹp đạo hàm ở mức ±20% (Bí quyết của PPO)
gamma = 0.99
lr = 0.002

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

print("Đang huấn luyện PPO. Quan sát sự ổn định tuyệt đối...")

# 3. Bộ nhớ đệm (Chờ gom đủ Batch)
class Memory:
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.is_terminals = [], []
    def clear(self):
        del self.states[:], self.actions[:], self.logprobs[:]
        del self.rewards[:], self.is_terminals[:]

memory = Memory()
timestep = 0
running_reward = 0

for episode in range(1, 1501):
    state, _ = env.reset()
    done = False
    ep_reward = 0
    
    while not done:
        timestep += 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad(): # Tắt đạo hàm lúc đi thu thập dữ liệu
            action_probs, _ = model(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        if done and ep_reward < 499: 
            reward = -10 # Phạt để học ranh giới sinh tử
            
        # Lưu vào nhật ký
        memory.states.append(state)
        memory.actions.append(action.item())
        memory.logprobs.append(dist.log_prob(action).item())
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        
        state = next_state
        ep_reward += reward
        
        # ==========================================
        # QUÁ TRÌNH CẬP NHẬT PPO (Sau mỗi 2000 bước)
        # ==========================================
        if timestep >= update_timestep:
            # A. Tính phần thưởng tích lũy (Returns)
            returns = []
            discounted_reward = 0
            for r, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
                if is_terminal: discounted_reward = 0
                discounted_reward = r + (gamma * discounted_reward)
                returns.insert(0, discounted_reward)
                
            returns = torch.tensor(returns, dtype=torch.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)
            
            # Chuyển đổi toàn bộ Batch thành Tensor
            old_states = torch.FloatTensor(np.array(memory.states))
            old_actions = torch.LongTensor(np.array(memory.actions))
            old_logprobs = torch.FloatTensor(np.array(memory.logprobs))
            
            # B. Học lặp lại K_epochs lần trên cùng 1 Batch
            for _ in range(K_epochs):
                probs, state_values = model(old_states)
                dist = Categorical(probs)
                new_logprobs = dist.log_prob(old_actions)
                entropy = dist.entropy()
                
                # Tính tỷ lệ Ratio (Mới / Cũ)
                ratios = torch.exp(new_logprobs - old_logprobs)
                
                # Tính Advantage
                advantages = returns - state_values.squeeze().detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # C. CƠ CHẾ CLIPPING THẦN THÁNH CỦA PPO
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(state_values.squeeze(), returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            memory.clear()
            timestep = 0
            
    running_reward += (ep_reward + (10 if ep_reward < 499 else 0)) # Bù lại phần trừ ảo
    
    if episode % 20 == 0:
        avg_reward = running_reward / 20
        print(f"Episode {episode} \t Điểm trung bình (20 ván): {avg_reward:.2f}")
        running_reward = 0
        if avg_reward >= 490:
            print("Đã giải quyết thành công môi trường!")
            torch.save(model.state_dict(), "ppo_best.pth")
            break