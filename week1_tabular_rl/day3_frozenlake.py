import gymnasium as gym
import time

# 1. Khởi tạo môi trường Hồ Băng
# map_name="4x4" là lưới 4x4. is_slippery=False nghĩa là mặt băng không trơn (đi đâu trúng đó)
# render_mode="human" để mở cửa sổ pop-up xem Agent di chuyển
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

# 2. Bắt đầu game (Reset môi trường về trạng thái ban đầu)
state, info = env.reset()
print(f"Trạng thái bắt đầu: {state}")

# Chơi thử 5 bước ngẫu nhiên
for step in range(20):
    print(f"\n--- Bước {step + 1} ---")
    
    # 3. Agent chọn một hành động ngẫu nhiên (0: Trái, 1: Xuống, 2: Phải, 3: Lên)
    action = env.action_space.sample() 
    print(f"Hành động Agent chọn: {action}")
    
    # 4. Gửi hành động vào môi trường và nhận lại phản hồi
    next_state, reward, terminated, truncated, info = env.step(action)
    
    print(f"Trạng thái mới: {next_state}")
    print(f"Phần thưởng nhận được: {reward}")
    
    # Nếu Agent rơi xuống hố (terminated) hoặc đi quá lâu (truncated) -> Game Over
    if terminated or truncated:
        print("Game Over! Đang reset lại game...")
        state, info = env.reset()
    else:
        state = next_state # Cập nhật trạng thái để chuẩn bị cho bước lặp sau
        
    time.sleep(1) # Dừng 1 giây để mắt người kịp nhìn màn hình pop-up

env.close()