import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import pathlib 
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import tyro
from torch.utils.tensorboard import SummaryWriter
import random
import time
import optax
import mujoco
from mujoco import mjx
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# 导入 Brax 相关模块
import brax
from brax import envs

# ── 自定义极简 Gymnasium 转换器 (避开 Brax 原生 Bug) ─────────────────────
class BraxToGymWrapper(gym.Env):
    """A robust wrapper to convert Brax MJX envs to Gymnasium."""
    def __init__(self, env: envs.Env, seed: int = 0):
        self._env = env
        self._key = jax.random.PRNGKey(seed)
        self._state = None
        
        if hasattr(self._env.sys, 'actuator_ctrlrange'):
            low = np.array(self._env.sys.actuator_ctrlrange[:, 0])
            high = np.array(self._env.sys.actuator_ctrlrange[:, 1])
        else:
            low = -np.ones(self._env.action_size, dtype=np.float32)
            high = np.ones(self._env.action_size, dtype=np.float32)
            
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        obs_size = self._env.observation_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # 核心优化：强制环境编译并在 CPU 上执行，彻底消除 GPU 跨设备通信延迟！
        self._step_fn = jax.jit(self._env.step, backend='cpu')
        self._reset_fn = jax.jit(self._env.reset, backend='cpu')

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, reset_key = jax.random.split(self._key)
        self._state = self._reset_fn(reset_key)
        # 此时数据本身就在 CPU 内存中，np.array 是零延迟的
        return np.array(self._state.obs), {}

    def step(self, action):
        self._state = self._step_fn(self._state, jax.numpy.array(action))
        obs = np.array(self._state.obs)
        reward = float(self._state.reward)
        done = bool(self._state.done)
        info = {k: np.array(v) for k, v in self._state.info.items()}
        
        truncated = bool(info.get('truncation', False))
        terminated = done and not truncated
        
        return obs, reward, terminated, truncated, info

from models import SimpleAgent, FinetuneAgent

# ── Constants ──────────────────────────────────────────────────────────
LOG_STD_MAX = 2
LOG_STD_MIN = -20

# ── Geom ↔ body-part mapping for HalfCheetah ───────────────────────────
_PART_TO_GEOMS = {
    'torso': ['torso'],
    'thigh': ['bthigh', 'fthigh'],
    'shin':  ['bshin',  'fshin'],
    'foot':  ['bfoot',  'ffoot'],
}

# ── Scenarios ──────────────────────────────────────────────────────────
SCENARIOS = {
    'forgetting':       ['hugefoot', 'moon', 'carry_stuff', 'rainfall'] * 2,
    'transfer':         ['carry_stuff_hugegravity', 'moon', 'defective_module',
                         'hugefoot_rainfall'] * 2,
    'robustness':       ['normal', 'inverted_actions', 'normal', 'inverted_actions'] * 2,
    'compositionality': ['tinyfoot', 'moon', 'carry_stuff_hugegravity', 'tinyfoot_moon'] * 2,
}

# ── Task definitions ───────────────────────────────────────────────────
TASK_MODS = {
    "normal": {},
    "carry_stuff": {"torso": 4., "thigh": 1., "shin": 1., "foot": 1.},
    'carry_stuff_hugegravity': {'torso': 4.0, 'thigh': 1.0, 'shin': 1.0,
                                'foot': 1.0, 'gravity': 1.5},
    "defective_module": {"obs_mask": 0.5},
    "hugefoot": {"foot": 1.5},
    "hugefoot_rainfall": {'foot': 1.5, 'friction': 0.4},
    "inverted_actions": {"action_swap": [0, 1, 2, 3, 4, 5]},
    "moon": {"gravity": 0.15},
    "tinyfoot": {"foot": 0.5},
    "tinyfoot_moon": {'foot': 0.5, 'gravity': 0.15},
    "rainfall": {"friction": 0.4},
}


# ── Args ───────────────────────────────────────────────────────────────
@dataclass
class Args:
    model_type: Literal['simple', 'finetune', 'sdnet'] = 'simple'
    """The name of the NN model to use for the agent"""
    task_name: Optional[str] = None
    """Single task name to run (ignored if --scenario is set)"""
    scenario: Optional[str] = None
    """Scenario name: forgetting, transfer, robustness, compositionality"""
    save_dir: Optional[str] = None
    """Directory to save trained models"""
    prev_dir: Tuple[pathlib.Path, ...] = ()
    """Directory containing previous-task models (for finetune / sdnet)"""
    exp_name: str = os.path.basename(__file__)[:-len('.py')]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    torch_deterministic: bool = True
    """If toggled, torch.backends.cudnn.deterministic=False (ignored in JAX)"""
    cuda: bool = True
    """If toggled, JAX will use GPU if available"""
    track: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = 'cw-sac'
    """the wandb's project name"""
    wandb_entity: Optional[str] = None  # 修复 Tyro 警告：使用 Optional 允许为 None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """Whether to capture videos of the agent performances"""

    # Algorithm specific arguments
    eval_every: int = 10_000
    """Evaluate the agent in deterministic mode every X timesteps"""
    num_evals: int = 10
    """Number of times to evaluate the agent"""
    total_timesteps: int = int(1e6)
    """total timesteps of the experiment"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discounted factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the replay memory"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    random_actions_end: int = 10_000
    """timesteps to take actions randomly"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


# ── Brax Environment Wrappers (JAX Native) ──────────────────────────────
class BraxObsMaskWrapper(envs.Wrapper):
    """JAX native wrapper for masking observations."""
    def __init__(self, env: envs.Env, mask_prob: float):
        super().__init__(env)
        self.mask_prob = mask_prob

    def reset(self, rng: jax.Array) -> envs.State:
        rng, reset_rng = jax.random.split(rng)
        state = self.env.reset(reset_rng)
        info = state.info
        info['mask_rng'] = rng
        state = state.replace(info=info)
        return self._mask_obs(state)

    def step(self, state: envs.State, action: jax.Array) -> envs.State:
        rng, step_rng = jax.random.split(state.info.get('mask_rng', jax.random.PRNGKey(0)))
        state = self.env.step(state, action)
        info = state.info
        info['mask_rng'] = rng
        state = state.replace(info=info)
        return self._mask_obs(state, step_rng)

    def _mask_obs(self, state: envs.State, rng: jax.Array = None) -> envs.State:
        if rng is None:
            rng = state.info.get('mask_rng', jax.random.PRNGKey(0))
        mask = jax.random.bernoulli(rng, 1.0 - self.mask_prob, shape=state.obs.shape).astype(state.obs.dtype)
        return state.replace(obs=state.obs * mask)


class BraxActionSwapWrapper(envs.Wrapper):
    """JAX native wrapper for swapping actions."""
    def __init__(self, env: envs.Env, perm: list):
        super().__init__(env)
        self.perm = jnp.array(perm, dtype=jnp.int32)

    def step(self, state: envs.State, action: jax.Array) -> envs.State:
        action = action[self.perm]
        return self.env.step(state, action)


# ── Environment factory ─────────────────────────────────────────────────
def make_env(task_name: str, seed: int = 0):
    """Create a Brax HalfCheetah env using MJX backend with specific modifications."""
    mods = TASK_MODS.get(task_name, {})

    xml_path = os.path.join(os.path.dirname(brax.envs.__file__), 'assets', 'half_cheetah.xml')
    if not os.path.exists(xml_path):
        import gymnasium.envs.mujoco
        xml_path = os.path.join(os.path.dirname(gymnasium.envs.mujoco.__file__), 'assets', 'half_cheetah.xml')
        
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    for key, scale in mods.items():
        if key in _PART_TO_GEOMS:
            for gname in _PART_TO_GEOMS[key]:
                gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gname)
                if gid >= 0:
                    mj_model.geom_size[gid, 0] *= scale
        elif key == 'gravity':
            mj_model.opt.gravity[2] = -9.81 * scale
        elif key == 'friction':
            if mj_model.ngeom > 0:
                mj_model.geom_friction[:, 0] *= scale

    # 核心优化：告诉 MJX 把物理模型放在 CPU 上
    cpu_device = jax.devices("cpu")[0]
    sys = mjx.put_model(mj_model, device=cpu_device)

    env = envs.create("halfcheetah", backend="mjx")
    env.sys = sys

    if 'obs_mask' in mods:
        env = BraxObsMaskWrapper(env, mods['obs_mask'])
    if 'action_swap' in mods:
        env = BraxActionSwapWrapper(env, mods['action_swap'])

    env = BraxToGymWrapper(env, seed=seed)
    return env


# ── Replay Buffer ──────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            'observations': jnp.array(self.obs[idxs]),
            'next_observations': jnp.array(self.next_obs[idxs]),
            'actions': jnp.array(self.actions[idxs]),
            'rewards': jnp.array(self.rewards[idxs]),
            'dones': jnp.array(self.dones[idxs]),
        }

    def __len__(self):
        return self.size


# ── Flax Networks ──────────────────────────────────────────────────────
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


# ── Action sampling (reparameterization trick) ─────────────────────────
def sample_action(model, params, obs, key, action_scale=1.0, action_bias=0.0,
                  **model_kwargs):
    """Sample an action from the Gaussian policy using reparameterization."""
    mean, log_std = model.apply(params, obs, **model_kwargs)
    log_std = jnp.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, shape=mean.shape)
    x_t = mean + std * noise
    y_t = jnp.tanh(x_t)
    action = y_t * action_scale + action_bias
    log_prob = jnp.sum(
        jax.scipy.stats.norm.logpdf(x_t, mean, std), axis=-1, keepdims=True
    )
    log_prob -= jnp.sum(
        jnp.log(action_scale * (1 - y_t ** 2) + 1e-6), axis=-1, keepdims=True
    )
    mean_action = jnp.tanh(mean) * action_scale + action_bias
    return action, log_prob, mean_action


# ── Evaluation ─────────────────────────────────────────────────────────
def eval_agent(actor_model, actor_params, env, num_evals, global_step,
               writer, action_scale=1.0, action_bias=0.0):
    obs, _ = env.reset()
    avg_ep_ret = 0.0
    ep_ret = 0.0
    rng = jax.random.PRNGKey(0)
    completed_episodes = 0

    while completed_episodes < num_evals:
        obs_jax = jnp.array(obs, dtype=jnp.float32)[None, ...]
        rng, rng_a = jax.random.split(rng)
        action, _, _ = sample_action(
            actor_model, actor_params, obs_jax, rng_a,
            action_scale=action_scale, action_bias=action_bias,
        )
        obs, reward, termination, truncation, info = env.step(np.array(action[0]))
        ep_ret += float(reward)

        if termination or truncation:
            avg_ep_ret += ep_ret
            completed_episodes += 1
            obs, _ = env.reset()
            ep_ret = 0.0

    avg_ep_ret /= num_evals
    print(f'\nTEST: ep_ret={avg_ep_ret:.3f}\n')
    writer.add_scalar("charts/test_episodic_return", avg_ep_ret, global_step)


# ── Run a single task ──────────────────────────────────────────────────
# ── Run a single task ──────────────────────────────────────────────────
def run_task(args: Args, task_name: str, prev_model_dir: Optional[str],
             scenario_name: str = ""):
    """Train SAC on a single HalfCheetah task. Returns the save directory."""
    
    # 动态识别：如果有场景名，就用场景名建文件夹；如果是 simple 并行，就放在 baseline 下
    scenario_prefix = f"{scenario_name}/" if scenario_name else "baseline/"
    run_name = f'{scenario_prefix}task_{task_name}__{args.model_type}__{args.exp_name}__{args.seed}'
    print(f'\n*** 进程启动: {run_name} ***\n')

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f'runs/{run_name}')
    
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(task_name, seed=args.seed)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    assert isinstance(env.action_space, spaces.Box), "only continuous action space is supported"

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_scale = float((env.action_space.high[0] - env.action_space.low[0]) / 2.0)
    action_bias = float((env.action_space.high[0] + env.action_space.low[0]) / 2.0)

    # Replay buffer
    rb = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, max_size=args.buffer_size)

    # Build model & initialise params
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_init = jax.random.split(rng)

    print(f'*** Loading model `{args.model_type}`')

    if args.model_type == 'simple':
        actor_model = SimpleAgent(act_dim=act_dim)
        actor_params = actor_model.init(rng_init, jnp.ones((1, obs_dim)))

    elif args.model_type == 'finetune':
        assert prev_model_dir is not None, (
            f'Model type {args.model_type} requires a previous model '
            f'(pass --prev-dir or use --scenario)'
        )
        actor_params = SimpleAgent.load_and_reset_heads(
            args.prev_dir[0], reset_heads=True, rng=rng_init, obs_dim=obs_dim, act_dim=act_dim
        )
        actor_model = SimpleAgent(act_dim=act_dim)

    elif args.model_type == 'sdnet':
        raise NotImplementedError(
            'sdnet (SingleNet) is not yet ported to JAX/Flax. '
            'Implement models/singlenet.py first.'
        )
    else:
        raise ValueError(f'Unknown model_type: {args.model_type}')

    # Q-networks
    qf1 = QNetwork()
    qf2 = QNetwork()
    rng, rng_q1, rng_q2 = jax.random.split(rng, 3)
    dummy_obs = jnp.ones((1, obs_dim))
    dummy_act = jnp.ones((1, act_dim))
    qf1_params = qf1.init(rng_q1, dummy_obs, dummy_act)
    qf2_params = qf2.init(rng_q2, dummy_obs, dummy_act)
    qf1_target_params = qf1_params
    qf2_target_params = qf2_params

    # Entropy tuning
    if args.autotune:
        target_entropy = -float(act_dim)
        log_alpha = jnp.zeros(1)
    else:
        target_entropy = None
        log_alpha = jnp.log(jnp.array(args.alpha))

    # Optimisers
    q_optimizer = optax.adam(args.q_lr)
    actor_optimizer = optax.adam(args.policy_lr)
    alpha_optimizer = optax.adam(args.q_lr)

    q_opt_state = q_optimizer.init((qf1_params, qf2_params))
    actor_opt_state = actor_optimizer.init(actor_params)
    alpha_opt_state = alpha_optimizer.init(log_alpha)

    # ── JIT-compiled SAC update ──────────────────────────────────────────
    @jax.jit
    def sac_update(rng, actor_params, qf1_params, qf2_params,
                   qf1_target_params, qf2_target_params, log_alpha,
                   q_opt_state, actor_opt_state, alpha_opt_state,
                   batch, update_policy):
        obs = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_observations']
        dones = batch['dones']

        rng, rng_next, rng_pi, rng_alpha = jax.random.split(rng, 4)

        # ── Q-function update ──────────────────────────────────────
        next_actions, next_log_pi, _ = sample_action(
            actor_model, actor_params, next_obs, rng_next,
            action_scale=action_scale, action_bias=action_bias,
        )
        qf1_next = qf1.apply(qf1_target_params, next_obs, next_actions)
        qf2_next = qf2.apply(qf2_target_params, next_obs, next_actions)
        min_qf_next = jnp.minimum(qf1_next, qf2_next)
        target_q = rewards + args.gamma * (1 - dones) * (
            min_qf_next - jnp.exp(log_alpha) * next_log_pi
        )

        def q_loss_fn(p1, p2):
            q1 = qf1.apply(p1, obs, actions)
            q2 = qf2.apply(p2, obs, actions)
            l1 = jnp.mean((q1 - jax.lax.stop_gradient(target_q)) ** 2)
            l2 = jnp.mean((q2 - jax.lax.stop_gradient(target_q)) ** 2)
            return l1 + l2, (l1, l2)

        (qf_loss, (qf1_loss, qf2_loss)), q_grads = jax.value_and_grad(
            q_loss_fn, argnums=(0, 1), has_aux=True
        )(qf1_params, qf2_params)

        q_updates, q_opt_state = q_optimizer.update(q_grads, q_opt_state)
        qf1_params = optax.apply_updates(qf1_params, q_updates[0])
        qf2_params = optax.apply_updates(qf2_params, q_updates[1])

        # ── Policy + alpha update (delayed) ────────────────────────
        def _no_update(ap, lp, aop, aap):
            return ap, lp, aop, aap, 0.0, 0.0

        def _do_update(ap, lp, aop, aap):
            def _one_iter(i, carry):
                ap, lp, aop, aap, k_pi, k_alpha = carry
                k_pi, k_pi_use = jax.random.split(k_pi)
                k_alpha, k_alpha_use = jax.random.split(k_alpha)

                # Actor loss
                def actor_loss_fn(p):
                    pi, log_pi, _ = sample_action(
                        actor_model, p, obs, k_pi_use,
                        action_scale=action_scale, action_bias=action_bias,
                    )
                    q1 = qf1.apply(qf1_params, obs, pi)
                    q2 = qf2.apply(qf2_params, obs, pi)
                    return jnp.mean(jnp.exp(lp) * log_pi - jnp.minimum(q1, q2))

                a_loss, a_grads = jax.value_and_grad(actor_loss_fn)(ap)
                a_updates, aop = actor_optimizer.update(a_grads, aop)
                ap = optax.apply_updates(ap, a_updates)

                # Alpha loss
                def alpha_loss_fn(la):
                    pi_a, log_pi_a, _ = sample_action(
                        actor_model, ap, obs, k_alpha_use,
                        action_scale=action_scale, action_bias=action_bias,
                    )
                    return -jnp.mean(
                        jnp.exp(la)
                        * (jax.lax.stop_gradient(log_pi_a) + target_entropy)
                    )

                al_loss, al_grads = jax.value_and_grad(alpha_loss_fn)(lp)
                al_updates, aap = alpha_optimizer.update(al_grads, aap)
                lp = optax.apply_updates(lp, al_updates)
                return ap, lp, aop, aap, k_pi, k_alpha

            ap, lp, aop, aap, _, _ = jax.lax.fori_loop(
                0, args.policy_frequency, _one_iter,
                (ap, lp, aop, aap, rng_pi, rng_alpha),
            )
            
            # Recompute losses for logging from final state
            pi_f, log_pi_f, _ = sample_action(
                actor_model, ap, obs, rng_pi,
                action_scale=action_scale, action_bias=action_bias,
            )
            q1_f = qf1.apply(qf1_params, obs, pi_f)
            q2_f = qf2.apply(qf2_params, obs, pi_f)
            a_loss_final = jnp.mean(
                jnp.exp(lp) * log_pi_f - jnp.minimum(q1_f, q2_f)
            )
            al_loss_final = -jnp.mean(
                jnp.exp(lp) * (jax.lax.stop_gradient(log_pi_f) + target_entropy)
            )
            return ap, lp, aop, aap, a_loss_final, al_loss_final

        (actor_params, log_alpha, actor_opt_state, alpha_opt_state,
         actor_loss, alpha_loss) = jax.lax.cond(
            update_policy, _do_update, _no_update,
            actor_params, log_alpha, actor_opt_state, alpha_opt_state,
        )

        # ── Target network update (polyak averaging) ────────────────
        qf1_target_params = jax.tree_util.tree_map(
            lambda t, s: args.tau * s + (1 - args.tau) * t,
            qf1_target_params, qf1_params,
        )
        qf2_target_params = jax.tree_util.tree_map(
            lambda t, s: args.tau * s + (1 - args.tau) * t,
            qf2_target_params, qf2_params,
        )

        info = {
            'qf1_loss': qf1_loss,
            'qf2_loss': qf2_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': jnp.exp(log_alpha),
            'qf1_mean': jnp.mean(qf1.apply(qf1_params, obs, actions)),
            'qf2_mean': jnp.mean(qf2.apply(qf2_params, obs, actions)),
        }
        return (rng, actor_params, qf1_params, qf2_params,
                qf1_target_params, qf2_target_params, log_alpha,
                q_opt_state, actor_opt_state, alpha_opt_state, info)

    # ── Training loop ──────────────────────────────────────────────────
    start_time = time.time()
    obs, _ = env.reset(seed=args.seed)
    episode_count = 0

    for global_step in range(args.total_timesteps):
        if global_step < args.random_actions_end:
            action = env.action_space.sample()
        else:
            rng, rng_act = jax.random.split(rng)
            act_jax, _, _ = sample_action(
                actor_model, actor_params,
                jnp.array(obs, dtype=jnp.float32)[None, ...],
                rng_act,
                action_scale=action_scale, action_bias=action_bias,
            )
            action = np.array(act_jax[0])

        next_obs, reward, termination, truncation, info = env.step(action)

        # Log episode returns to TensorBoard
        # Log episode returns to TensorBoard
        if 'episode' in info:
            # 修复 Numpy 标量转换警告
            ep_r = info['episode']['r'].item() if hasattr(info['episode']['r'], 'item') else float(info['episode']['r'])
            ep_l = info['episode']['l'].item() if hasattr(info['episode']['l'], 'item') else int(info['episode']['l'])
            print(f"global_step={global_step}, episodic_return={ep_r:.3f}, episodic_length={ep_l}")
            writer.add_scalar('charts/episodic_return', float(ep_r), global_step)
            writer.add_scalar('charts/episodic_length', int(ep_l), global_step)

        real_next_obs = next_obs.copy()
        if truncation and 'final_observation' in info:
            real_next_obs = info['final_observation']

        rb.add(obs, real_next_obs, action, reward, float(termination))
        obs = next_obs

        # Periodic SPS / loss logging
        in_random_phase = global_step < args.random_actions_end
        log_interval = 1000 if in_random_phase else 500
        if global_step % log_interval == 0:
            sps = int(global_step / (time.time() - start_time))
            if not in_random_phase and global_step > args.learning_starts:
                print(f"SPS={sps}")
            elif in_random_phase:
                print(f"SPS={sps}")

        if global_step > args.learning_starts:
            batch = rb.sample(args.batch_size)
            do_update_policy = (global_step % args.policy_frequency == 0)

            (rng, actor_params, qf1_params, qf2_params,
             qf1_target_params, qf2_target_params, log_alpha,
             q_opt_state, actor_opt_state, alpha_opt_state,
             info_dict) = sac_update(
                rng=rng,
                actor_params=actor_params,
                qf1_params=qf1_params,
                qf2_params=qf2_params,
                qf1_target_params=qf1_target_params,
                qf2_target_params=qf2_target_params,
                log_alpha=log_alpha,
                q_opt_state=q_opt_state,
                actor_opt_state=actor_opt_state,
                alpha_opt_state=alpha_opt_state,
                batch=batch,
                update_policy=do_update_policy,
            )

            if global_step % log_interval == 0:
                for k, v in info_dict.items():
                    # 兼容 () 和 (1,) 形状，安全提取出纯数字标量
                    val = v.item() if hasattr(v, 'item') else float(v)
                    writer.add_scalar(f'losses/{k}', val, global_step)
                writer.add_scalar('charts/SPS', sps, global_step)
                writer.add_scalar('charts/global_step', global_step, global_step)

        if global_step % args.eval_every == 0 and global_step > 0:
            eval_agent(actor_model, actor_params, env, args.num_evals,
                       global_step, writer,
                       action_scale=action_scale, action_bias=action_bias)

    # Final evaluation
    eval_agent(actor_model, actor_params, env, args.num_evals,
               args.total_timesteps, writer,
               action_scale=action_scale, action_bias=action_bias)

    env.close()
    writer.close()

    # Save
    save_path = f'{args.save_dir}/{run_name}'
    print(f'Saving trained agent to `{save_path}`')
    actor_model.save(save_path, actor_params)
    return save_path


import copy
import pathlib

# ── Helper for Parallel Continual Learning ──────────────────────────────
def run_scenario_worker(args: Args, scenario_name: str, task_sequence: list):
    """在一个独立的进程中，按顺序跑完某个场景下的所有任务，实现知识传递。"""
    print(f'\n🚀 [子进程启动] 开始评估场景: 【{scenario_name}】')
    
    # 💥 使用深拷贝，防止 4 个并行场景互相覆盖彼此的 args 参数
    local_args = copy.deepcopy(args)
    
    # 获取根保存路径（如你例子中的 "agents" 文件夹）
    base_save_dir = local_args.save_dir if local_args.save_dir else "agents"
    
    # ── 逻辑 1：锚定 Task 0 的 Simple Baseline 权重 ──
    first_task = task_sequence[0]
    # 严格按照 run_task 生成的命名规则，拼凑出 Task 0 对应的 simple 模型路径
    prev_dir = f"{base_save_dir}/baseline/task_{first_task}__simple__{local_args.exp_name}__{local_args.seed}"
    
    print(f'--- 场景 [{scenario_name}] | 初始基线准备完毕 ---')
    print(f'==> 已锚定 Task 0 [{first_task}] 的基础权重: {prev_dir}')

    # ── 逻辑 2：从 Task 1 开始真正的 Finetune 串行循环 ──
    for i, task_name in enumerate(task_sequence[1:], start=1):
        print(f'\n--- 场景 [{scenario_name}] | 持续学习: 任务 {i}/{len(task_sequence)-1}: {task_name} ---')
        
        # ── 逻辑 3：完美适配你改版的 prev_dir (Tuple[pathlib.Path, ...]) ──
        # 将上一轮的路径包装成 Path 对象并放入元组中，供 run_task 里的 args.prev_dir[0] 调用
        local_args.prev_dir = (pathlib.Path(prev_dir),)
        print(f'==> 继承权重路径: {local_args.prev_dir[0]}')
        
        # 运行当前任务。跑完后，新生成的 finetune 模型路径会覆盖 prev_dir，自动传给下一次循环！
        prev_dir = run_task(local_args, task_name, prev_model_dir=prev_dir, scenario_name=scenario_name)
        
    print(f'\n✅ [子进程完成] 场景 【{scenario_name}】 评估彻底结束！\n')
    return scenario_name


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 强制设置 Python 多进程使用 'spawn' 模式，防止 JAX/CUDA 显存互串
    multiprocessing.set_start_method('spawn', force=True)
    
    args = tyro.cli(Args)

    # 限制最大并发进程数，防止显卡 OOM（你提到的 4 个进程是目前最稳妥的）
    MAX_WORKERS = 4 

    # ── 逻辑分流 1：Simple 模型 (11个任务的独立并行排队，自动 4+4+3) ──
    if args.model_type == 'simple':
        tasks_to_run = list(TASK_MODS.keys())
        
        print(f'\n{"=" * 60}')
        print(f'[Parallel Baseline Mode] 模型类别为 `{args.model_type}`。')
        print(f'即将将以下 {len(tasks_to_run)} 个独立任务丢入多进程池排队运行：')
        print(f'{tasks_to_run}')
        print(f'最大并发度: {MAX_WORKERS}')
        print(f'{"=" * 60}\n')
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for task_name in tasks_to_run:
                # 提交给线程池。它会自动维护 4 个在跑，跑完一个自动补下一个
                futures.append(executor.submit(run_task, args, task_name, None, "baseline"))
            
            for future in futures:
                future.result()  # 阻塞并抛出可能的报错
                
        print(f'\n🎉 所有 {len(tasks_to_run)} 个 `simple` 基线任务已全部跑完！\n')

    # ── 逻辑分流 2：持续学习模型 (4大场景各自拉起独立模型，同时并行) ──
    elif args.model_type == 'finetune':
        scenarios_to_run = list(SCENARIOS.keys())
        
        print(f'\n{"=" * 60}')
        print(f'[Parallel Scenario Mode] 模型类别为 `{args.model_type}`。')
        print(f'即将为您拉起 {len(scenarios_to_run)} 个独立进程，各自跑通一条场景链。')
        print(f'场景列表: {scenarios_to_run}')
        print(f'{"=" * 60}\n')
        
        # 因为刚好有 4 个场景，所以会瞬间拉起 4 个进程同时干活
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for scenario_name in scenarios_to_run:
                task_sequence = SCENARIOS[scenario_name]
                # 提交场景级 Worker
                futures.append(executor.submit(run_scenario_worker, args, scenario_name, task_sequence))
            
            for future in futures:
                future.result()
                
        print(f'\n🎉 持续学习网络：所有场景的评估链路已全部执行完毕！\n')