import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer

# ==========================================
# 1. 定义 3 层 MLP 网络
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        # 3层全连接：输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh() # 假设动作空间是 [-1, 1] 的连续控制任务
        )

    def forward(self, obs):
        return self.net(obs)

# ==========================================
# 2. 初始化环境、网络和 Buffer
# ==========================================
def collect_experiences(task_name="Pendulum-v1", total_steps=100):
    # 初始化环境
    env = gym.make(task_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 初始化 3 层 MLP
    agent = SimpleMLP(obs_dim, act_dim)
    agent.eval() # 设置为评估模式（不启用 Dropout 等训练特性）

    # 初始化 ReplayBuffer (假设单环境运行，容量设置为大于等于收集步数即可)
    buffer = ReplayBuffer(
        buffer_size=total_steps,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device="cpu", # 不涉及训练，放 CPU 即可
        n_envs=1
    )

    # ==========================================
    # 3. 与环境交互并存入 Buffer
    # ==========================================
    obs, info = env.reset()
    
    print(f"开始与 {task_name} 环境交互收集数据...")
    
    for step in range(total_steps):
        # 将 numpy 数组转为 Tensor，并增加 batch_size 维度：[obs_dim] -> [1, obs_dim]
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        # 不计算梯度，加速推理并节省内存
        with torch.no_grad():
            action_tensor = agent(obs_tensor)
            
        # 将 Tensor 转回 numpy 数组，并去掉 batch_size 维度：[1, act_dim] -> [act_dim]
        action = action_tensor.squeeze(0).numpy()
        
        # 与环境交互一步
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 将这条经验存入 Buffer
        buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            infos=[info] # SB3 要求的格式
        )
        
        # 如果环境结束（比如失败或者达到最大步数），则重置环境
        if done:
            obs, info = env.reset()
        else:
            obs = next_obs
            
    # 交互结束
    print(f"成功收集了 {buffer.pos} 步数据！")
    print(f"目前 Buffer 中的状态数据维度为: {buffer.observations.shape}")
    
    env.close()
    return buffer

def inspect_buffer_data(buffer, index=0):
    """
    查看 ReplayBuffer 中指定索引的数据。
    
    参数:
        buffer: Stable Baselines3 的 ReplayBuffer 对象
        index: 你想查看的数据的索引（从 0 开始）
    """
    # 1. 安全检查：确保你想看的数据没有超出当前存入的范围
    # buffer.full 表示池子是否存满了（如果满了，有效数据就是整个 buffer_size；没满就是 buffer.pos）
    max_valid_idx = buffer.buffer_size if buffer.full else buffer.pos
    
    if index >= max_valid_idx:
        print(f"❌ 索引越界！你请求了第 {index} 条，但当前 Buffer 中只有 {max_valid_idx} 条有效数据。")
        return
        
    # 2. 提取数据
    # 注意：因为 SB3 支持并行环境，底层数组多了一个维度 n_envs。
    # 所以我们用 [index, 0] 来取出第 index 步、第 0 个环境的数据。
    obs = buffer.observations[index, 0]
    action = buffer.actions[index, 0]
    reward = buffer.rewards[index, 0]
    done = buffer.dones[index, 0]
    next_obs = buffer.next_observations[index, 0]

    # 3. 打印展示结果
    print(f"========== Buffer 数据详情 (索引: {index}) ==========")
    print(f"🔹 状态 (Observation)  维度 {obs.shape}:\n{obs}")
    print(f"🔹 动作 (Action)       维度 {action.shape}:\n{action}")
    print(f"🔹 奖励 (Reward):      {reward}")
    print(f"🔹 结束标志 (Done):    {bool(done)}")
    # 如果状态太长，不想全部打印，可以用下面这行注释掉 next_obs 的打印
    # print(f"🔹 下一状态 (Next Obs) 维度 {next_obs.shape}:\n{next_obs}")
    print("====================================================")
    
def sampling(buffer: ReplayBuffer, k_samples):
    samples = buffer.sample(k_samples)

    obs = samples.observations
    act = samples.actions

    return obs, act



# ==========================================
# 4. 执行主程序
# ==========================================
if __name__ == "__main__":
    # 你可以把这里替换成你需要的 Gymnasium 或 Meta-World 任务名称
    my_buffer = collect_experiences(task_name="Pendulum-v1", total_steps=100)
    # =======================
    # 测试调用示例
    # 假设你在前面的代码中返回了 my_buffer
    # 查看第 1 条（最开始的）数据
    # inspect_buffer_data(my_buffer, index=0)

    # # 查看第 50 条数据
    # inspect_buffer_data(my_buffer, index=49)

    # # 查看最新存入的一条数据
    # latest_index = my_buffer.pos - 1
    # inspect_buffer_data(my_buffer, index=latest_index)
    obs, act = sampling(my_buffer, 3)
    print(f'observation is \n{obs}')
    print(f'action is \n{act}')

    obs = obs.unsqueeze(0)
    act = act.unsqueeze(0)

    print(f'***After concat')
    print(f'matrix of obs is \n{obs}')
    print(f'matrix of act is \n{act}')