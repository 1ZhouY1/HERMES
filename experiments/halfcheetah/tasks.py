import gymnasium as gym
import numpy as np
import mujoco

# ── 1. 官方开源项目原版任务字典 ──
env_tasks = {
    "normal": {},
    "carry_stuff": {"torso": 4., "thigh": 1., "shin": 1., "foot": 1.},
    "carry_stuff_hugegravity": {'torso': 4.0, 'thigh': 1.0, 'shin': 1.0, 'foot': 1.0, 'gravity': 1.5},
    "defective_module": {"obs_mask": 0.5},
    "hugefoot": {"foot": 1.5},
    "hugefoot_rainfall": {'foot': 1.5, 'friction': 0.4},
    "inverted_actions": {"action_swap": [0,1,2,3,4,5]},
    "moon": {"gravity": 0.15},
    "tinyfoot": {"foot": 0.5},
    "tinyfoot_moon": {'foot': 0.5, 'gravity': 0.15},
    "rainfall": {"friction": 0.4},
}

# 自动生成不重复的任务列表，长度刚好为 11。
# 在 SAC.py 中传入的 task_id (0~10) 就会一一对应到这里。
tasks = list(env_tasks.keys())

# ── 2. 基于官方字典键名的四大场景序列 ──
SCENARIOS = {
    "forgetting": ["hugefoot", "moon", "carry_stuff", "rainfall"] * 2,
    "transfer": ["carry_stuff_hugegravity", "moon", "defective_module", "hugefoot_rainfall"] * 2,
    "robustness": ["normal", "inverted_actions", "normal", "inverted_actions"] * 2,
    "compositionality": ["tinyfoot", "moon", "carry_stuff_hugegravity", "tinyfoot_moon"] * 2
}

def get_task_name(task_id):
    return tasks[task_id]


# ── 3. 定义特殊任务所需的 Gym Wrappers ──

class HalfCheetahCompatibilityWrapper(gym.Wrapper):
    """注入 success 字段以兼容 Meta-World 训练代码"""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['success'] = 0.0  # 半猎豹没有绝对意义的成功，默认记为 0
        return obs, reward, terminated, truncated, info

class InvertedActionWrapper(gym.ActionWrapper):
    """反转指定的动作维度，对应 inverted_actions"""
    def __init__(self, env, swap_axes):
        super().__init__(env)
        self.swap_axes = swap_axes

    def action(self, action):
        action = action.copy()
        action[self.swap_axes] = -action[self.swap_axes]
        return action

class ObsMaskWrapper(gym.ObservationWrapper):
    """遮蔽部分观测值，对应 defective_module"""
    def __init__(self, env, mask_ratio):
        super().__init__(env)
        obs_len = env.observation_space.shape[0]
        num_mask = int(obs_len * mask_ratio)
        # 固定遮蔽掉后 half 的观测数据模拟传感器损坏
        self.mask = np.ones(obs_len, dtype=np.float32)
        self.mask[-num_mask:] = 0.0 

    def observation(self, observation):
        return observation * self.mask


# ── 4. 核心：动态解析字典并生成物理环境 ──

def get_task(task_id, render=False):
    task_name = get_task_name(task_id)
    render_mode = "human" if render else None
    
    # 获取该任务的具体配置字典
    mods = env_tasks[task_name]

    # 创建标准环境 (要求 Gymnasium >= 0.29.0)
    env = gym.make('HalfCheetah-v4', render_mode=render_mode)
    model = env.unwrapped.model

    # 1) 动态处理身体部位的 质量 与 几何体大小
    part_to_names = {
        "torso": ["torso"],
        "thigh": ["bthigh", "fthigh"],
        "shin": ["bshin", "fshin"],
        "foot": ["bfoot", "ffoot"]
    }
    
    for part, names in part_to_names.items():
        if part in mods:
            scale = mods[part]
            for name in names:
                # 修改质量 (安全写法：检查 ID 是否 >= 0，避开 -1 隐患)
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    model.body_mass[bid] *= scale
                
                # 修改几何体大小 (安全写法：检查 ID 是否 >= 0)
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    model.geom_size[gid, 0] *= scale

    # 2) 动态处理全局重力 (moon, hugegravity)
    if 'gravity' in mods:
        model.opt.gravity[2] = -9.81 * mods['gravity']

    # 3) 动态处理全局摩擦力 (rainfall)
    if 'friction' in mods:
        if model.ngeom > 0:
            model.geom_friction[:, 0] *= mods['friction']

    # 4) 处理动作反转 Wrapper (inverted_actions)
    if 'action_swap' in mods:
        env = InvertedActionWrapper(env, mods['action_swap'])
        
    # 5) 处理传感器损坏 Wrapper (defective_module)
    if 'obs_mask' in mods:
        env = ObsMaskWrapper(env, mods['obs_mask'])

    # 最后套上 Meta-World 兼容层
    env = HalfCheetahCompatibilityWrapper(env)
    
    return env


if __name__ == '__main__':
    # 本地测试：遍历打印 11 个任务，确保一切运转正常
    for i in range(len(tasks)):
        env = get_task(i)
        print(f"✅ 成功加载任务 [{i}]: {get_task_name(i):<25} | 参数: {env_tasks[get_task_name(i)]}")