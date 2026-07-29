import torch
import torch.nn as nn
import numpy as np
from tasks import get_task
import gymnasium as gym
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import pathlib
import os
import json
import re
import tyro
from torch.utils.tensorboard import SummaryWriter
import random
import time
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from models import SingleNetAgent, shared, SimpleAgent, GRUMLPAgent, GRUNoContextAgent, StateGRUContextAgent, RecurrentSoftQNetwork, sample_gru_replay_batch, sample_state_gru_replay_batch, LongTermMemory, HybridSoftQNetwork, HybridRAGRLAgent, GRUStateEncoder, SummaryMemory, SummaryMemoryAgent, FrozenBackboneReplaceHeadAgent, FrozenSimpleResidualAdapterAgent, PairMemory, RewardReplacingPairMemory, RARLAgent, AdaptiveRARLAgent, HybridRARLAgent, ReservoirTrajectoryMemory
import torch.nn.functional as F


@dataclass
class Args:
    model_type: Literal['simple', 'finetune', 'replace-head', 'residual-adapter', 'packnet', 'prognet', 'sdnet', 'grumlp', 'gnet', 'state-gru', 'state-gru-traj50', 'only-sdnet', 'ragnet', 'ragnet-wo-context', 'ragnet-wo-a', 'ragnet-concat-only', 'hybridrarl', 'hybridrarl-wo-gate', 'hybrid_add_norm', 'rarl', "rarl-500", "rarl-2000", "rarl-2000-adaptive", "rarl-5000", "rarl-10000", 'rarl-reward', 'rarl-reward-500', 'rarl-reward-ctx', 'hybrid-encoder', 'hybrid-context', 'hybrid-raw-encoder', 'hybrid-raw', 'hybrid-raw-context', 'hybrid', 'gate', 'summary-memory']
    """The name of the NN model to use for the agent"""
    save_dir: Optional[str] = None
    """If provided, the model will be saved in the given directory"""
    save_every: int = 0
    """Save an intermediate checkpoint every N environment steps. 0 disables periodic saving"""
    save_memory: bool = False
    """If toggled, save LongTermMemory together with agents that support it"""
    prev_units: Tuple[pathlib.Path, ...] = ()
    """Paths to the previous models. Not required when model_type is `simple` or `packnet` or `prognet` or `sdnet`"""
    exp_name: str = os.path.basename(__file__)[: -len('.py')]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    torch_deterministic: bool = True
    """If toggled, torch.backends.cudnn.deterministic=False"""
    cuda: bool = True
    """If toggled, cuda will be enabled by default"""
    track: bool = False
    """If toggled, this experiments will be tracked with Weights and Biases"""
    wandb_project_name: str = 'cw-sac'
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """Wheather to capture videos of the agent performances (check out `videos` folder)"""


    # Algorithm specific arguments
    task_id: int = 0
    """ID name of the task"""
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
    gru_seq_len: int = 20
    """number of previous (observation, action, reward) transitions used for GRU context"""
    gru_hidden_dim: int = 64
    """hidden size for the single-layer actor/critic GRUs"""
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

    ragnet_n_traj: int = 64
    traj_len: int = 50
    ragnet_traj_len: int = 50
    """the trajectory number and length for the RAGNet's long term memory"""

    # for Hybrid RAGRL
    hr_n_traj: int = 512
    """retrieve batch size: number of trajectories sampled from long-term memory"""
    hr_traj_len: int = 50
    """fixed trajectory length stored in Hybrid RAGRL memory"""
    hr_top_r: int = 8
    """coarse retrieval: number of top trajectories selected from retrieve batch"""
    hr_top_k_exact: int = 16
    """fine retrieval: number of state/action candidates selected inside top-r trajectories"""
    hr_state_dim: int = 128
    """encoded state dimension for Hybrid RAGRL actor and critic"""
    hr_traj_summary_dim: int = 256
    """trajectory summary dimension for coarse semantic retrieval"""

    # for Summary Memory policy
    sm_memory_capacity: int = 2000
    """maximum number of trajectory summary vectors stored in SummaryMemory"""
    sm_traj_len: int = 50
    """trajectory length used to write one SummaryMemory item"""
    sm_summary_dim: int = 128
    """GRU hidden dimension for trajectory summaries"""
    sm_context_dim: int = 128
    """attention context dimension retrieved from SummaryMemory"""
    sm_attention_dim: int = 128
    """query/key attention dimension for SummaryMemory retrieval"""
    sm_similarity_threshold: float = 0.95
    """cosine similarity threshold used when deciding summary replacement"""
    sm_pretrained_summary: Optional[str] = None
    """Optional checkpoint from train_summary_action.py used to initialize the Summary encoder"""
    sm_freeze_summary: bool = True
    """Freeze the pretrained Summary encoder during RL interaction/training"""

    adapter_hidden_dim: int = 256
    """Hidden dimension for the residual-adapter model type"""

    # for RARL auxiliary losses
    rarl_aux_coef: float = 0.1
    """coefficient for RARL summarizer auxiliary loss"""

    rarl_aux_reward_coef: float = 1.0
    """coefficient for reward prediction auxiliary loss"""

    rarl_aux_return_coef: float = 1.0
    """coefficient for return-to-go prediction auxiliary loss"""

    rarl_aux_action_coef: float = 0.1
    """coefficient for action prediction auxiliary loss"""

    rarl_kl_coef: float = 1e-3
    """coefficient for RARL information bottleneck KL loss"""

    rarl_memory_size: int = 1000
    """number of state-action pairs stored in RARL FIFO memory"""

    rarl_att_dim: int = 128
    """query/key attention dimension for pair-memory RARL"""

    rarl_value_dim: int = 128
    """value/context dimension retrieved from pair-memory RARL"""

    rarl_replace_neighbors: int = 50
    """number of nearest states considered by rarl-reward replacement"""

    adaptive_warmup_steps: int = 20_000
    """random state samples collected before rarl-2000-adaptive training; not counted in total_timesteps"""

    adaptive_similarity_threshold: float = 0.7
    """reuse a previous policy prefix when the nearest state Wasserstein distance is below this value"""

    adaptive_state_num_projections: int = 512
    """number of random projections used by rarl-2000-adaptive sliced Wasserstein"""

    adaptive_state_file: str = "sampled_states.pt"
    """filename used to save the pre-training random state sequence in each agent directory"""

    adaptive_state_dir: Optional[pathlib.Path] = None
    """optional shared state-cache directory or filename template; cached task states are reused before random sampling"""

    adaptive_reuse_warmup_states: bool = True
    """reuse an existing adaptive_state_file instead of collecting it again"""

    adaptive_freeze_reused_prefix: bool = True
    """freeze copied policy prefix layers when a similar previous task is selected"""

    # for HybridRARL
    hybridrarl_memory_capacity: int = 100
    """maximum number of fixed-length trajectories kept per task"""

    hybridrarl_traj_len: int = 20
    """number of state-action pairs cached before writing one trajectory to memory"""

    hybridrarl_top_r: int = 5
    """coarse retrieval keeps the top-R trajectories"""

    hybridrarl_hidden_dim: int = 128
    """hidden dimension of the trajectory RNN and critic state feature"""

    hybridrarl_att_dim: int = 128
    """query/key attention dimension in fine retrieval"""

    hybridrarl_value_dim: int = 128
    """retrieved context dimension z_t"""

    hybridrarl_policy_hidden_dim: int = 256
    """hidden dimension of HybridRARL policy layers"""

    hybridrarl_policy_layers: int = 4
    """number of hidden Linear layers in HybridRARL policy trunk"""

    hybridrarl_coarse_metric: Literal["cosine", "l2"] = "cosine"
    """coarse retrieval metric between current state and trajectory first states"""

    hybridrarl_warmup_steps: int = 20_000
    """random states sampled before HybridRARL training; not counted in total_timesteps"""

    hybridrarl_similarity_threshold: float = 0.7
    """reuse previous HybridRARL modules when nearest state Wasserstein distance is below this value"""

    hybridrarl_state_num_projections: int = 512
    """number of random projections used by HybridRARL sliced Wasserstein"""

    hybridrarl_state_file: str = "hybridrarl_sampled_states.pt"
    """filename used to save HybridRARL pre-training random state samples"""

    hybridrarl_state_dir: Optional[pathlib.Path] = None
    """optional shared state-cache directory or filename template"""

    hybridrarl_reuse_warmup_states: bool = True
    """reuse existing HybridRARL sampled states when available"""

    hybridrarl_freeze_reused_modules: bool = True
    """freeze copied trajectory RNN and first two policy layers"""

def make_env(task_id):
    def thunk():
        env = get_task(task_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

class SoftContextQNetwork(nn.Module):
    def __init__(self, envs, context):
        super().__init__()
        self.fc = shared(
            np.array(envs.observation_space.shape).prod()
            + np.prod(envs.action_space.shape)
            + context.shape
        )

class SoftQNetwork(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super().__init__()

        # Make sure Linear receives a plain Python int, not np.ndarray / np.int64.
        if isinstance(feature_dim, np.ndarray):
            feature_dim = feature_dim.item()
        if isinstance(act_dim, np.ndarray):
            act_dim = act_dim.item()

        feature_dim = int(feature_dim)
        act_dim = int(act_dim)

        self.fc = shared(feature_dim + act_dim)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc(x)
        x = self.fc_out(x)
        return x



LOG_STD_MAX = 2
LOG_STD_MIN = -20



class Actor(nn.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv, model: nn.Module):
        super().__init__()
        self.model = model

        #action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32
            ),
        )

    def forward(self, x, **kwargs):
        return_hidden = kwargs.get("return_hidden", False)
        model_out = self.model(x, **kwargs)
        if return_hidden:
            mean, log_std, h = model_out
        else:
            mean, log_std = model_out
            h = None
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) # From SpinUp / Dennis Yarats

        if return_hidden:
            return mean, log_std, h

        return mean, log_std
    
    def get_action(self, x, **kwargs):
        return_hidden = kwargs.get("return_hidden", False)
        forward_out = self(x, **kwargs)
        if return_hidden:
            mean, log_std, h = forward_out
        else:
            mean, log_std = forward_out
            h = None
        std = log_std.exp() # exp(log_x) = x
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # for reparameterization trick (mean + std * sample ~ N(mean, std), where sample ~ N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        #Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) 
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        if return_hidden:
            return action, log_prob, mean, h
        return action, log_prob, mean

@torch.no_grad()
def eval_agent(agent, test_env, num_evals, global_step, writer, device):
    obs, _ = test_env.reset()
    avg_ep_ret = 0
    avg_success = 0
    ep_ret = 0
    gru_hidden = None
    gru_cache = []
    state_gru_cache = []
    summary_cache = []
    for _ in range(num_evals):
        while True:
            obs_np = obs
            obs = torch.Tensor(obs_np).to(device).unsqueeze(0)
            if isinstance(agent.model, (GRUMLPAgent, GRUNoContextAgent)):
                gru_history = build_gru_history_sequence(
                    history_cache=gru_cache,
                    obs_dim=agent.model.obs_dim,
                    act_dim=agent.model.act_dim,
                    history_len=getattr(agent.model, "context_seq_len", 20),
                    device=device,
                )
                _, _, action = agent.get_action(obs, history=gru_history)
            elif isinstance(agent.model, StateGRUContextAgent):
                state_sequence = build_state_gru_sequence(
                    state_cache=state_gru_cache,
                    current_obs=obs_np,
                    obs_dim=agent.model.obs_dim,
                    sequence_len=getattr(agent.model, "context_seq_len", 20),
                    device=device,
                )
                _, _, action = agent.get_action(obs, state_sequence=state_sequence)
            elif isinstance(agent.model, SummaryMemoryAgent):
                summary_cache.append(obs.squeeze(0).detach().cpu().numpy())
                summary_cache = summary_cache[-agent.model.traj_len :]
                summary_sequence = torch.as_tensor(
                    np.stack(summary_cache, axis=0),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                _, _, action = agent.get_action(
                    obs,
                    summary_sequence=summary_sequence,
                )
            else:
                _, _, action = agent.get_action(obs)
            action_np = action[0].cpu().numpy()
            obs, reward, termination, truncation, info = test_env.step(action_np)

            if isinstance(agent.model, (GRUMLPAgent, GRUNoContextAgent)):
                gru_cache.append(make_gru_history_item(obs_np, action_np, reward))
                gru_cache[:] = gru_cache[-int(getattr(agent.model, "context_seq_len", 20)) :]
            if isinstance(agent.model, StateGRUContextAgent):
                state_gru_cache.append(np.asarray(obs_np, dtype=np.float32).reshape(-1))
                state_gru_cache[:] = state_gru_cache[
                    -max(int(getattr(agent.model, "context_seq_len", 20)) - 1, 0) :
                ]

            ep_ret += reward

            if termination or truncation:
                print(f'info is \n{info}')
                avg_success += info['success']
                avg_ep_ret += ep_ret
                #reset
                obs, _ = test_env.reset()
                ep_ret = 0
                gru_hidden = None
                gru_cache.clear()
                state_gru_cache.clear()
                summary_cache.clear()
                break
    avg_ep_ret /= num_evals
    avg_success /= num_evals
    print(f'\nTEST: ep_ret={avg_ep_ret}, success={avg_success}\n')
    writer.add_scalar("charts/test_episodic_return", avg_ep_ret, global_step)
    writer.add_scalar("charts/test_success", avg_success, global_step)

def sample_ragnet_memory_batch(longterm_memory, n_traj, device):
    """
    Sample a retrieve batch from LongTermMemory.

    If current memory size is smaller than requested n_traj, use all available
    trajectories. This is important during early training. For example, after
    random_actions_end=10000 and traj_len=50, memory only contains about 200
    trajectories, which is smaller than hr_n_traj=512.
    """
    assert longterm_memory is not None, "longterm_memory is None."
    assert len(longterm_memory) > 0, "LongTermMemory is empty."

    effective_n_traj = min(int(n_traj), len(longterm_memory))

    traj_batch = longterm_memory.sample_trajectories(
        n_traj=effective_n_traj,
        device=device,
        replace=False,
    )

    return {
        "obs": traj_batch.states,
        "states": traj_batch.states,
        "actions": traj_batch.actions,
        "rewards": traj_batch.rewards,
    }


def save_agent(model, dirname, save_memory=False):
    if isinstance(model, SummaryMemoryAgent):
        model.save(dirname=dirname, save_memory=True)
        return

    if save_memory and hasattr(model, "memory"):
        model.save(dirname=dirname, save_memory=True)
    else:
        model.save(dirname=dirname)


def _serializable_args(args):
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, pathlib.Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = tuple(str(item) if isinstance(item, pathlib.Path) else item for item in value)
        else:
            result[key] = value
    return result


def save_sac_checkpoint(
    dirname,
    actor,
    qf1,
    qf2,
    qf1_target,
    qf2_target,
    actor_optimizer,
    q_optimizer,
    args,
    global_step,
    alpha,
    log_alpha=None,
    a_optimizer=None,
    save_memory=False,
):
    os.makedirs(dirname, exist_ok=True)
    save_agent(actor.model, dirname=dirname, save_memory=save_memory)

    checkpoint = {
        "type": "sac_full_checkpoint",
        "model_type": args.model_type,
        "global_step": int(global_step),
        "args": _serializable_args(args),
        "actor_state_dict": actor.state_dict(),
        "actor_model_state_dict": actor.model.state_dict(),
        "qf1_state_dict": qf1.state_dict(),
        "qf2_state_dict": qf2.state_dict(),
        "qf1_target_state_dict": qf1_target.state_dict(),
        "qf2_target_state_dict": qf2_target.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "alpha": float(alpha),
    }

    if hasattr(qf1, "gru"):
        checkpoint["critic_gru_state_dict"] = qf1.gru.state_dict()
        checkpoint["critic_target_gru_state_dict"] = qf1_target.gru.state_dict()
        checkpoint["critic_gru_shared_by_qf1_qf2"] = qf1.gru is qf2.gru
        checkpoint["critic_target_gru_shared_by_qf1_qf2"] = qf1_target.gru is qf2_target.gru

    if hasattr(actor.model, "gru"):
        checkpoint["actor_gru_state_dict"] = actor.model.gru.state_dict()

    if log_alpha is not None:
        checkpoint["log_alpha"] = log_alpha.detach().cpu()
    if a_optimizer is not None:
        checkpoint["alpha_optimizer_state_dict"] = a_optimizer.state_dict()

    torch.save(checkpoint, os.path.join(dirname, "sac_checkpoint.pt"))


def longterm_capacity_from_transition_budget(buffer_size, traj_len):
    return max(1, int(buffer_size) // int(traj_len))


def build_summary_sequence(cache_states, current_obs, traj_len, device):
    current_state = np.asarray(current_obs, dtype=np.float32).reshape(-1)
    recent_states = cache_states[-max(int(traj_len) - 1, 0) :] + [current_state]
    return torch.as_tensor(
        np.stack(recent_states, axis=0),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def build_gru_history_sequence(history_cache, obs_dim, act_dim, history_len, device):
    history_dim = int(obs_dim) + int(act_dim) + 1
    history = np.zeros((int(history_len), history_dim), dtype=np.float32)
    recent_history = history_cache[-int(history_len) :]
    if len(recent_history) > 0:
        history[-len(recent_history) :] = np.stack(recent_history, axis=0)

    return torch.as_tensor(
        history,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def build_state_gru_sequence(state_cache, current_obs, obs_dim, sequence_len, device):
    sequence = np.zeros((int(sequence_len), int(obs_dim)), dtype=np.float32)
    current_state = np.asarray(current_obs, dtype=np.float32).reshape(-1)
    recent_states = state_cache[-max(int(sequence_len) - 1, 0) :] + [current_state]
    sequence[-len(recent_states) :] = np.stack(recent_states, axis=0)

    return torch.as_tensor(
        sequence,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def state_gru_sequence_len(args):
    return 50 if args.model_type == "state-gru-traj50" else int(args.gru_seq_len)


def make_gru_history_item(obs, action, reward):
    return np.concatenate(
        [
            np.asarray(obs, dtype=np.float32).reshape(-1),
            np.asarray(action, dtype=np.float32).reshape(-1),
            np.asarray([reward], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )


def unique_parameters(*modules):
    params = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            param_id = id(param)
            if param_id not in seen:
                params.append(param)
                seen.add(param_id)
    return params


def soft_update_modules(source_modules, target_modules, tau):
    seen_targets = set()
    for source_module, target_module in zip(source_modules, target_modules):
        for param, target_param in zip(
            source_module.parameters(),
            target_module.parameters(),
        ):
            target_id = id(target_param)
            if target_id in seen_targets:
                continue
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
            seen_targets.add(target_id)


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_task_id_from_path(path):
    match = re.search(r"task_(\d+)__", pathlib.Path(path).name)
    if match is None:
        raise ValueError(f"Cannot infer task id from path: {path}")
    return int(match.group(1))


def reset_plain_env(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_plain_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def collect_random_state_sequence(task_id, num_steps, seed):
    env = get_task(task_id)
    obs = reset_plain_env(env, seed=seed)
    states = []
    episode_returns = []
    episode_successes = []
    ep_return = 0.0
    ep_success = 0.0

    for step in range(int(num_steps)):
        states.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        action = np.asarray(env.action_space.sample(), dtype=np.float32).reshape(-1)
        obs, reward, done, info = step_plain_env(env, action)
        ep_return += float(reward)
        ep_success = max(ep_success, float(info.get("success", 0.0)))

        if done:
            episode_returns.append(ep_return)
            episode_successes.append(ep_success)
            obs = reset_plain_env(env, seed=seed + step + 1)
            ep_return = 0.0
            ep_success = 0.0

    if ep_return != 0.0 or not episode_returns:
        episode_returns.append(ep_return)
        episode_successes.append(ep_success)

    env.close()
    states = np.stack(states, axis=0).astype(np.float32)
    meta = {
        "task_id": int(task_id),
        "num_states": int(states.shape[0]),
        "state_dim": int(states.shape[1]),
        "seed": int(seed),
        "random_sampling_steps": int(num_steps),
        "mean_random_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_random_success": float(np.mean(episode_successes)) if episode_successes else 0.0,
    }
    return states, meta


def save_state_sequence(path, states, meta):
    os.makedirs(path.parent, exist_ok=True)
    torch.save(
        {
            "states": torch.as_tensor(states, dtype=torch.float32),
            "meta": meta,
        },
        path,
    )


def load_state_sequence(path):
    payload = safe_torch_load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        states = payload.detach().cpu().numpy()
        meta = {}
    elif isinstance(payload, dict):
        key = "states"
        if key not in payload and "data" in payload:
            data = payload["data"]
            key = "states" if "states" in data else "observations"
            states = data[key]
        else:
            states = payload[key]
        meta = payload.get("meta", {})
        states = states.detach().cpu().numpy() if isinstance(states, torch.Tensor) else np.asarray(states)
    else:
        raise TypeError(f"Unsupported sampled state file payload: {type(payload)}")

    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    if states.ndim != 2:
        raise ValueError(f"Expected 2D states in {path}, got {states.shape}")
    return states, meta


def resolve_state_sequence_path(model_dir, task_id, state_file, state_dir=None):
    if state_dir is None:
        return pathlib.Path(model_dir) / state_file

    state_dir_text = str(state_dir)
    if "{task_id" in state_dir_text or "{task_id02}" in state_dir_text:
        return pathlib.Path(
            state_dir_text.format(
                task_id=int(task_id),
                task_id02=f"{int(task_id):02d}",
            )
        )

    base = pathlib.Path(state_dir)
    if base.suffix == ".pt":
        return base

    candidates = [
        base / f"task_{int(task_id)}__{state_file}",
        base / f"task_{int(task_id):02d}__{state_file}",
        base / f"task_{int(task_id)}_states.pt",
        base / f"task_{int(task_id):02d}_states.pt",
        base / f"target_task_{int(task_id)}.pt",
        base / f"target_task_{int(task_id):02d}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def ensure_state_sequence(
    model_dir,
    task_id,
    state_file,
    num_steps,
    seed,
    reuse_existing=True,
    state_dir=None,
):
    path = resolve_state_sequence_path(
        model_dir=model_dir,
        task_id=task_id,
        state_file=state_file,
        state_dir=state_dir,
    )
    if reuse_existing and path.exists():
        states, meta = load_state_sequence(path)
        print(f"*** Reusing cached states for task {task_id}: `{path}`")
        return states, meta, path

    print(
        f"*** Collecting {num_steps} random states for task {task_id} "
        f"and saving to `{path}` (not counted in SAC total_timesteps)"
    )
    states, meta = collect_random_state_sequence(
        task_id=task_id,
        num_steps=num_steps,
        seed=seed,
    )
    save_state_sequence(path, states, meta)
    return states, meta, path


def normalize_state_sets(state_sets):
    all_states = np.concatenate(list(state_sets.values()), axis=0)
    mean = all_states.mean(axis=0, keepdims=True)
    std = np.maximum(all_states.std(axis=0, keepdims=True), 1e-6)
    return {
        key: ((states - mean) / std).astype(np.float32)
        for key, states in state_sets.items()
    }


def sliced_wasserstein_distance_np(x, y, num_projections, seed):
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"state dim mismatch: {x.shape[1]} vs {y.shape[1]}")
    rng = np.random.default_rng(seed)
    n = min(x.shape[0], y.shape[0])
    if x.shape[0] != n:
        x = x[rng.choice(x.shape[0], size=n, replace=False)]
    if y.shape[0] != n:
        y = y[rng.choice(y.shape[0], size=n, replace=False)]

    directions = rng.normal(size=(int(num_projections), x.shape[1])).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-8)
    x_proj = np.sort(x @ directions.T, axis=0)
    y_proj = np.sort(y @ directions.T, axis=0)
    return float(np.sqrt(np.mean((x_proj - y_proj) ** 2)))


def select_adaptive_reuse_source(
    current_states,
    prev_units,
    state_file,
    state_dir,
    warmup_steps,
    seed,
    num_projections,
    threshold,
    reuse_existing=True,
):
    if len(prev_units) == 0:
        return None, []

    state_sets = {"current": current_states}
    prev_state_info = []
    for index, prev_dir in enumerate(prev_units):
        prev_dir = pathlib.Path(prev_dir)
        prev_task_id = parse_task_id_from_path(prev_dir)
        prev_states, prev_meta, prev_state_path = ensure_state_sequence(
            model_dir=prev_dir,
            task_id=prev_task_id,
            state_file=state_file,
            num_steps=warmup_steps,
            seed=seed + prev_task_id * 10000,
            reuse_existing=reuse_existing,
            state_dir=state_dir,
        )
        key = f"prev_{index}"
        state_sets[key] = prev_states
        prev_state_info.append(
            {
                "key": key,
                "task_id": int(prev_task_id),
                "model_dir": str(prev_dir),
                "state_file": str(prev_state_path),
                "num_states": int(prev_states.shape[0]),
                "meta": prev_meta,
            }
        )

    normalized_sets = normalize_state_sets(state_sets)
    distances = []
    for index, info in enumerate(prev_state_info):
        distance = sliced_wasserstein_distance_np(
            normalized_sets["current"],
            normalized_sets[info["key"]],
            num_projections=num_projections,
            seed=seed + index * 1009,
        )
        item = {
            "task_id": info["task_id"],
            "model_dir": info["model_dir"],
            "state_file": info["state_file"],
            "distance": distance,
            "is_similar": bool(distance < threshold),
        }
        distances.append(item)
        print(
            f"*** Adaptive state distance: current vs task {info['task_id']} "
            f"= {distance:.6f} (threshold={threshold})"
        )

    similar = [item for item in distances if item["is_similar"]]
    if not similar:
        return None, distances
    return min(similar, key=lambda item: item["distance"]), distances


def load_pair_memory_file(path):
    data = safe_torch_load(path, map_location="cpu")
    memory_cls = RewardReplacingPairMemory if "rewards" in data else PairMemory
    kwargs = {
        "capacity": data["capacity"],
        "obs_dim": data["obs_dim"],
        "action_dim": data["action_dim"],
        "device": "cpu",
        "dtype": data.get("dtype", torch.float32),
    }
    if memory_cls is RewardReplacingPairMemory:
        kwargs["replace_neighbors"] = data.get("replace_neighbors", 50)

    memory = memory_cls(**kwargs)
    memory.ptr = int(data["ptr"])
    memory.size = int(data["size"])
    memory.states.copy_(data["states"])
    memory.actions.copy_(data["actions"])
    if hasattr(memory, "rewards") and "rewards" in data:
        memory.rewards.copy_(data["rewards"])
        memory.num_replacements = int(data.get("num_replacements", 0))
        memory.num_skips = int(data.get("num_skips", 0))
    return memory


def attach_pair_memory_if_needed(model, model_dir):
    if not hasattr(model, "retrieve_context"):
        return
    if getattr(model, "memory", None) is not None:
        return

    for filename in ["reward_replacing_pair_memory.pt", "pair_memory.pt"]:
        memory_path = pathlib.Path(model_dir) / filename
        if memory_path.exists():
            model.memory = load_pair_memory_file(memory_path)
            return


def attach_adaptive_prev_units_if_needed(model, map_location):
    if not isinstance(model, AdaptiveRARLAgent):
        return
    prev_units = [
        load_policy_unit(pathlib.Path(path), map_location=map_location)
        for path in getattr(model, "prev_unit_paths", [])
        if pathlib.Path(path).exists()
    ]
    model.set_prev_units(prev_units)


def load_policy_unit(model_dir, map_location):
    model_path = pathlib.Path(model_dir) / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No model.pt found in previous unit dir: {model_dir}")
    model = safe_torch_load(model_path, map_location=map_location)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module in {model_path}, got {type(model)}")
    attach_pair_memory_if_needed(model, model_dir)
    attach_adaptive_prev_units_if_needed(model, map_location=map_location)
    model = model.to(map_location).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


if __name__ == '__main__':
    # envs = gym.vector.SyncVectorEnv([make_env(0)]) # The SyncVectorEnv method requires a list of function as input
    # high = envs.single_action_space.high
    # low = envs.single_action_space.low
    # print(f'high: {high}') # [1,1,1,1]
    # print(f'low: {low}') # [-1,-1,-1,-1]

    args = tyro.cli(Args)

    ragnet_types = [
    "ragnet",
    "ragnet-wo-context",
    "ragnet-wo-a",
    "ragnet-concat-only",
    ]

    hybridrarl_types = [
        "hybridrarl",
    ]

    hybrid_types = [
        "hybridrarl-wo-gate",
        "hybrid_add_norm",
        "hybrid-encoder",
        "hybrid-context",
    ]

    raw_types = [
        "hybrid-raw-encoder",
        "hybrid-raw-context",
        "hybrid-raw",
        "hybrid",
        "gate",
    ]

    summary_memory_types = [
        "summary-memory",
    ]

    rarl_types = [
        "rarl",
        "rarl-500",
        "rarl-2000",
        "rarl-2000-adaptive",
        "rarl-5000",
        "rarl-10000",
        "rarl-reward",
        "rarl-reward-500",
        "rarl-reward-ctx",
    ]

    memory_types = ragnet_types + hybrid_types + hybridrarl_types
    fusion_mode_by_model_type = {
        "gate": "mixture",
        "hybrid": "residual",
    }

    run_name = f'task_{args.task_id}__{args.model_type}__{args.exp_name}__{args.seed}'
    print(f'\n*** Run name: {run_name} ***\n')
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
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f'*** Device {device}')

    adaptive_agent_dir = None
    adaptive_current_states = None
    adaptive_reuse_source = None
    adaptive_distance_info = []
    hybridrarl_agent_dir = None
    hybridrarl_current_states = None
    hybridrarl_reuse_source = None
    hybridrarl_distance_info = []
    if args.model_type == "rarl-2000-adaptive":
        if args.save_dir is None:
            raise ValueError("rarl-2000-adaptive requires --save-dir so sampled states can be saved.")

        adaptive_agent_dir = pathlib.Path(args.save_dir) / run_name
        adaptive_current_states, adaptive_current_meta, adaptive_state_path = ensure_state_sequence(
            model_dir=adaptive_agent_dir,
            task_id=args.task_id,
            state_file=args.adaptive_state_file,
            num_steps=args.adaptive_warmup_steps,
            seed=args.seed + args.task_id * 10000,
            reuse_existing=args.adaptive_reuse_warmup_states,
            state_dir=args.adaptive_state_dir,
        )
        adaptive_reuse_source, adaptive_distance_info = select_adaptive_reuse_source(
            current_states=adaptive_current_states,
            prev_units=args.prev_units,
            state_file=args.adaptive_state_file,
            state_dir=args.adaptive_state_dir,
            warmup_steps=args.adaptive_warmup_steps,
            seed=args.seed,
            num_projections=args.adaptive_state_num_projections,
            threshold=args.adaptive_similarity_threshold,
            reuse_existing=args.adaptive_reuse_warmup_states,
        )
        adaptive_meta_path = adaptive_agent_dir / "adaptive_reuse.json"
        adaptive_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with adaptive_meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_id": int(args.task_id),
                    "state_file": str(adaptive_state_path),
                    "state_dir": None if args.adaptive_state_dir is None else str(args.adaptive_state_dir),
                    "warmup_steps": int(args.adaptive_warmup_steps),
                    "similarity_threshold": float(args.adaptive_similarity_threshold),
                    "num_projections": int(args.adaptive_state_num_projections),
                    "selected_reuse_source": adaptive_reuse_source,
                    "distances": adaptive_distance_info,
                    "current_state_meta": adaptive_current_meta,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        if adaptive_reuse_source is None:
            print("*** Adaptive RARL: no previous task below threshold; policy prefix is newly initialized.")
        else:
            print(
                "*** Adaptive RARL: reusing policy prefix from "
                f"task {adaptive_reuse_source['task_id']} "
                f"(distance={adaptive_reuse_source['distance']:.6f})."
            )
    elif args.model_type == "hybridrarl":
        if args.save_dir is None:
            raise ValueError("hybridrarl requires --save-dir so sampled states and reuse metadata can be saved.")

        hybridrarl_agent_dir = pathlib.Path(args.save_dir) / run_name
        hybridrarl_current_states, hybridrarl_current_meta, hybridrarl_state_path = ensure_state_sequence(
            model_dir=hybridrarl_agent_dir,
            task_id=args.task_id,
            state_file=args.hybridrarl_state_file,
            num_steps=args.hybridrarl_warmup_steps,
            seed=args.seed + args.task_id * 10000,
            reuse_existing=args.hybridrarl_reuse_warmup_states,
            state_dir=args.hybridrarl_state_dir,
        )
        hybridrarl_reuse_source, hybridrarl_distance_info = select_adaptive_reuse_source(
            current_states=hybridrarl_current_states,
            prev_units=args.prev_units,
            state_file=args.hybridrarl_state_file,
            state_dir=args.hybridrarl_state_dir,
            warmup_steps=args.hybridrarl_warmup_steps,
            seed=args.seed,
            num_projections=args.hybridrarl_state_num_projections,
            threshold=args.hybridrarl_similarity_threshold,
            reuse_existing=args.hybridrarl_reuse_warmup_states,
        )
        hybridrarl_meta_path = hybridrarl_agent_dir / "hybridrarl_reuse.json"
        hybridrarl_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with hybridrarl_meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_id": int(args.task_id),
                    "state_file": str(hybridrarl_state_path),
                    "state_dir": None if args.hybridrarl_state_dir is None else str(args.hybridrarl_state_dir),
                    "warmup_steps": int(args.hybridrarl_warmup_steps),
                    "similarity_threshold": float(args.hybridrarl_similarity_threshold),
                    "num_projections": int(args.hybridrarl_state_num_projections),
                    "selected_reuse_source": hybridrarl_reuse_source,
                    "distances": hybridrarl_distance_info,
                    "current_state_meta": hybridrarl_current_meta,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        if hybridrarl_reuse_source is None:
            print("*** HybridRARL: no previous task below threshold; all modules are newly initialized.")
        else:
            print(
                "*** HybridRARL: reusing trajectory RNN and policy prefix from "
                f"task {hybridrarl_reuse_source['task_id']} "
                f"(distance={hybridrarl_reuse_source['distance']:.6f})."
            )

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.task_id)])
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    #select the model to use as the agent
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    print(f'*** Loading model `{args.model_type}` ***')

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    longterm_memory = None
    summary_memory = None
    cache_states = []
    cache_actions = []
    cache_rewards = []

    if args.model_type in ['finetune', 'replace-head', 'residual-adapter']:
        assert(
            len(args.prev_units) > 0
        ), f'Model type {args.model_type} requires at least one previous unit'
    
    if args.model_type == 'simple':
        model = SimpleAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)
    elif args.model_type == 'grumlp':
        model = GRUMLPAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=args.gru_hidden_dim,
            gru_layers=1,
        ).to(device)
        model.context_seq_len = args.gru_seq_len
    elif args.model_type == 'gnet':
        model = GRUNoContextAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=args.gru_hidden_dim,
            gru_layers=1,
        ).to(device)
        model.context_seq_len = args.gru_seq_len
    elif args.model_type in ['state-gru', 'state-gru-traj50']:
        model = StateGRUContextAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=args.gru_hidden_dim,
            gru_layers=1,
        ).to(device)
        model.context_seq_len = state_gru_sequence_len(args)
    elif args.model_type == 'finetune':
        model = SimpleAgent.load(
            args.prev_units[0], map_location=device, reset_heads=True
        ).to(device)
    elif args.model_type == 'replace-head':
        model = FrozenBackboneReplaceHeadAgent.load_from_simple(
            args.prev_units[0],
            map_location=device,
        ).to(device)
    elif args.model_type == 'residual-adapter':
        model = FrozenSimpleResidualAdapterAgent.load_from_simple(
            args.prev_units[0],
            map_location=device,
            adapter_hidden_dim=args.adapter_hidden_dim,
        ).to(device)
    elif args.model_type in ['ragnet', 'ragnet-wo-context', 'ragnet-wo-a', 'ragnet-concat-only']:
        longterm_capacity = longterm_capacity_from_transition_budget(
            args.buffer_size,
            args.ragnet_traj_len,
        )
        print(
            f"*** LongTermMemory capacity: {longterm_capacity} trajectories "
            f"({longterm_capacity * args.ragnet_traj_len} transitions)"
        )

        longterm_memory = LongTermMemory(
            capacity=longterm_capacity,
            traj_len=args.ragnet_traj_len,
            obs_dim=obs_dim,
            action_dim=act_dim,
            reward_dim=1,
            device=device,
        )

        model = RAGNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            state_dim=128,
            summary_dim=256,
            context_dim=128,
            hidden_dim=256,
            top_k=10,
        ).to(device)

    elif args.model_type in hybridrarl_types:
        print(
            f"*** HybridRARL ReservoirTrajectoryMemory capacity: "
            f"{args.hybridrarl_memory_capacity} trajectories "
            f"({args.hybridrarl_memory_capacity * args.hybridrarl_traj_len} transitions)"
        )

        longterm_memory = ReservoirTrajectoryMemory(
            capacity=args.hybridrarl_memory_capacity,
            traj_len=args.hybridrarl_traj_len,
            obs_dim=obs_dim,
            action_dim=act_dim,
            reward_dim=1,
            device="cpu",
        )

        model = HybridRARLAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            memory=longterm_memory,
            memory_capacity=args.hybridrarl_memory_capacity,
            traj_len=args.hybridrarl_traj_len,
            top_r=args.hybridrarl_top_r,
            traj_hidden_dim=args.hybridrarl_hidden_dim,
            att_dim=args.hybridrarl_att_dim,
            value_dim=args.hybridrarl_value_dim,
            policy_hidden_dim=args.hybridrarl_policy_hidden_dim,
            policy_layers=args.hybridrarl_policy_layers,
            coarse_metric=args.hybridrarl_coarse_metric,
        ).to(device)

        if hybridrarl_reuse_source is not None:
            source_model = load_policy_unit(
                pathlib.Path(hybridrarl_reuse_source["model_dir"]),
                map_location=device,
            )
            model.copy_transfer_modules_from(
                source_model,
                freeze=args.hybridrarl_freeze_reused_modules,
            )
            model.reused_source = hybridrarl_reuse_source["model_dir"]

    elif args.model_type in hybrid_types + raw_types:
        longterm_capacity = longterm_capacity_from_transition_budget(
            args.buffer_size,
            args.hr_traj_len,
        )
        print(
            f"*** LongTermMemory capacity: {longterm_capacity} trajectories "
            f"({longterm_capacity * args.hr_traj_len} transitions)"
        )

        longterm_memory = LongTermMemory(
            capacity=longterm_capacity,
            traj_len=args.hr_traj_len,
            obs_dim=obs_dim,
            action_dim=act_dim,
            reward_dim=1,
            device='cpu',
        )

        model = HybridRAGRLAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            memory=longterm_memory,
            n_traj=args.hr_n_traj,
            state_dim=args.hr_state_dim,
            traj_summary_dim=args.hr_traj_summary_dim,
            hidden_dim=256,
            top_r=args.hr_top_r,
            top_k_exact=args.hr_top_k_exact,
            fusion_mode=fusion_mode_by_model_type.get(args.model_type, "residual"),
        ).to(device)

    elif args.model_type in summary_memory_types:
        print(
            f"*** SummaryMemory capacity: {args.sm_memory_capacity} summaries "
            f"(traj_len={args.sm_traj_len})"
        )

        summary_memory = SummaryMemory(
            capacity=args.sm_memory_capacity,
            summary_dim=args.sm_summary_dim,
            similarity_threshold=args.sm_similarity_threshold,
            device="cpu",
        )

        model = SummaryMemoryAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            memory=summary_memory,
            summary_dim=args.sm_summary_dim,
            context_dim=args.sm_context_dim,
            hidden_dim=256,
            att_dim=args.sm_attention_dim,
            traj_len=args.sm_traj_len,
        ).to(device)

        if args.sm_pretrained_summary is not None:
            print(f"*** Loading pretrained Summary: {args.sm_pretrained_summary}")
            model.load_pretrained_summary(
                checkpoint_path=args.sm_pretrained_summary,
                freeze=args.sm_freeze_summary,
                map_location=device,
            )
    
    elif args.model_type in rarl_types:
        if args.model_type in ["rarl-reward", "rarl-reward-500", "rarl-reward-ctx"]:
            rarl_capacity = 500 if args.model_type == "rarl-reward-500" else args.rarl_memory_size
            print(
                f"*** RewardReplacingPairMemory capacity: {rarl_capacity} "
                f"state-action pairs (neighbors={args.rarl_replace_neighbors})"
            )
            longterm_memory = RewardReplacingPairMemory(
                capacity=rarl_capacity,
                obs_dim=obs_dim,
                action_dim=act_dim,
                replace_neighbors=args.rarl_replace_neighbors,
                device="cpu",
            )
        else:
            if args.model_type in ["rarl-500"]:
                print(
                f"*** PairMemory capacity: {args.rarl_memory_size} state-action pairs"
                )
                longterm_memory = PairMemory(
                    capacity=500,
                    obs_dim=obs_dim,
                    action_dim=act_dim,
                    device="cpu",
                )
            elif args.model_type in ["rarl-2000"]:
                print(
                f"*** PairMemory capacity: {args.rarl_memory_size} state-action pairs"
                )
                longterm_memory = PairMemory(
                    capacity=2000,
                    obs_dim=obs_dim,
                    action_dim=act_dim,
                    device="cpu",
                )
            elif args.model_type in ["rarl-2000-adaptive"]:
                print("*** PairMemory capacity: 2000 state-action pairs")
                longterm_memory = PairMemory(
                    capacity=2000,
                    obs_dim=obs_dim,
                    action_dim=act_dim,
                    device="cpu",
                )
            elif args.model_type in ["rarl-5000"]:
                print(
                f"*** PairMemory capacity: {args.rarl_memory_size} state-action pairs"
                )
                longterm_memory = PairMemory(
                    capacity=5000,
                    obs_dim=obs_dim,
                    action_dim=act_dim,
                    device="cpu",
                )
            elif args.model_type in ["rarl-10000"]:
                print(
                f"*** PairMemory capacity: {args.rarl_memory_size} state-action pairs"
                )
                longterm_memory = PairMemory(
                    capacity=10000,
                    obs_dim=obs_dim,
                    action_dim=act_dim,
                    device="cpu",
                )
            else:
                print(
                    f"*** PairMemory capacity: {args.rarl_memory_size} state-action pairs"
                )
                longterm_memory = PairMemory(
                    capacity=args.rarl_memory_size,
                    obs_dim=obs_dim,
                    action_dim=act_dim,
                    device="cpu",
                )

        if args.model_type == "rarl-2000-adaptive":
            prev_unit_models = [
                load_policy_unit(pathlib.Path(path), map_location=device)
                for path in args.prev_units
            ]
            model = AdaptiveRARLAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                memory=longterm_memory,
                prev_units=prev_unit_models,
                prev_unit_paths=args.prev_units,
                att_dim=args.rarl_att_dim,
                value_dim=args.rarl_value_dim,
                prev_att_dim=args.rarl_att_dim,
            ).to(device)
            if adaptive_reuse_source is not None:
                source_model = load_policy_unit(
                    pathlib.Path(adaptive_reuse_source["model_dir"]),
                    map_location=device,
                )
                model.copy_policy_prefix_from(
                    source_model,
                    freeze=args.adaptive_freeze_reused_prefix,
                )
                model.reused_policy_source = adaptive_reuse_source["model_dir"]
        else:
            model = RARLAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                memory=longterm_memory,
                att_dim=args.rarl_att_dim,
                value_dim=args.rarl_value_dim,
            ).to(device)

    elif args.model_type in ['sdnet', 'sdnet-rapid', 'only-sdnet']:
        model = SingleNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            buffer=rb,
            map_location=device,
        ).to(device)
    # Ignore the ProgNet and PackNet methods

    actor = Actor(envs, model).to(device)
    if args.model_type in hybridrarl_types:
        critic_feature_dim = obs_dim
    elif args.model_type in hybrid_types:
        critic_feature_dim = int(args.hr_state_dim)
    elif args.model_type in ["ragnet", "ragnet-wo-context", "ragnet-wo-a", "ragnet-concat-only"]:
        critic_feature_dim = actor.model.net_mean.policy_input_dim,
    elif args.model_type == "rarl-reward-ctx":
        critic_feature_dim = obs_dim + args.rarl_value_dim
    else:
        critic_feature_dim = obs_dim

    # critic_feature_dim = critic_feature_dim[0]
    # obs_dim = np.array(envs.single_observation_space.shape).prod()
    # act_dim = np.prod(envs.single_action_space.shape)
    print("critic_feature_dim =", critic_feature_dim, type(critic_feature_dim))
    print("act_dim =", act_dim, type(act_dim))
    print("critic input dim =", critic_feature_dim + act_dim, type(critic_feature_dim + act_dim))

    if args.model_type == "grumlp":
        critic_gru = nn.GRU(
            input_size=int(obs_dim) + int(act_dim) + 1,
            hidden_size=args.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        ).to(device)
        critic_target_gru = nn.GRU(
            input_size=int(obs_dim) + int(act_dim) + 1,
            hidden_size=args.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        ).to(device)
        qf1 = RecurrentSoftQNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=args.gru_hidden_dim,
            gru_layers=1,
            gru=critic_gru,
        ).to(device)
        qf2 = RecurrentSoftQNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=args.gru_hidden_dim,
            gru_layers=1,
            gru=critic_gru,
        ).to(device)
        qf1_target = RecurrentSoftQNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=args.gru_hidden_dim,
            gru_layers=1,
            gru=critic_target_gru,
        ).to(device)
        qf2_target = RecurrentSoftQNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=args.gru_hidden_dim,
            gru_layers=1,
            gru=critic_target_gru,
        ).to(device)
    else:
        qf1 = SoftQNetwork(critic_feature_dim, act_dim).to(device)
        qf2 = SoftQNetwork(critic_feature_dim, act_dim).to(device)
        qf1_target = SoftQNetwork(critic_feature_dim, act_dim).to(device)
        qf2_target = SoftQNetwork(critic_feature_dim, act_dim).to(device)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(unique_parameters(qf1, qf2), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.action_space.shape).to(device)
        ).item() 
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    
    start_time = time.time()
    cache_states = []
    cache_actions = []
    cache_rewards = []
    gru_history_cache = []
    state_gru_cache = []
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    gru_hidden = None
    for global_step in range(args.total_timesteps):
        
        # ALGO LOGIC: put action logic here
        if global_step < args.random_actions_end:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            obs_tensor = torch.Tensor(obs).to(device)

            if args.model_type in hybridrarl_types + hybrid_types + raw_types:
                assert len(longterm_memory) > 0, (
                    "Hybrid RAGRL starts acting after random_actions_end, but LongTermMemory is empty. "
                    "Increase random_actions_end or decrease the trajectory length."
                )

                retrieve_n_traj = (
                    args.hybridrarl_memory_capacity
                    if args.model_type in hybridrarl_types
                    else args.hr_n_traj
                )
                memory_batch = sample_ragnet_memory_batch(
                    longterm_memory=longterm_memory,
                    n_traj=retrieve_n_traj,
                    device=device,
                )

                with torch.no_grad():
                    actions, _, _ = actor.get_action(
                        obs_tensor,
                        memory_batch=memory_batch,
                    )
            
            elif args.model_type in rarl_types:
                assert len(longterm_memory) > 0, (
                    "RARL starts acting after random_actions_end, but PairMemory is empty. "
                    "Increase random_actions_end or check RARL pair-memory writes."
                )

                with torch.no_grad():
                    actions, _, _ = actor.get_action(obs_tensor)

            elif args.model_type in summary_memory_types:
                summary_sequence = build_summary_sequence(
                    cache_states=cache_states,
                    current_obs=obs[0],
                    traj_len=args.sm_traj_len,
                    device=device,
                )

                with torch.no_grad():
                    actions, _, _ = actor.get_action(
                        obs_tensor,
                        summary_sequence=summary_sequence,
                    )

            elif args.model_type in ["ragnet", "ragnet-wo-context", 'ragnet-wo-a', 'ragnet-concat-only']:
                assert len(longterm_memory) > 0, (
                    "RAGNet starts acting after random_actions_end, but LongTermMemory is empty. "
                    "Increase random_actions_end or decrease ragnet_traj_len."
                )

                memory_batch = sample_ragnet_memory_batch(
                    longterm_memory=longterm_memory,
                    n_traj=args.ragnet_n_traj,
                    device=device,
                )

                if global_step % 1000 == 0:
                    with torch.no_grad():
                        actions, _, _ = actor.get_action(
                            obs_tensor,
                            memory_batch=memory_batch,
                            writer=writer,
                            global_step=global_step,
                        )
                else:
                    with torch.no_grad():
                        actions, _, _ = actor.get_action(
                            obs_tensor,
                            memory_batch=memory_batch,
                        )

            elif args.model_type in ["component", 'sdnet', 'sdnet-rapid'] and global_step % 1000 == 0:
                with torch.no_grad():
                    actions, _, _ = actor.get_action(
                        obs_tensor,
                        writer=writer,
                        global_step=global_step,
                    )

            elif args.model_type in ["grumlp", "gnet"]:
                gru_history = build_gru_history_sequence(
                    history_cache=gru_history_cache,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    history_len=args.gru_seq_len,
                    device=device,
                )
                with torch.no_grad():
                    actions, _, _ = actor.get_action(obs_tensor, history=gru_history)
            elif args.model_type in ["state-gru", "state-gru-traj50"]:
                state_sequence = build_state_gru_sequence(
                    state_cache=state_gru_cache,
                    current_obs=obs[0],
                    obs_dim=obs_dim,
                    sequence_len=state_gru_sequence_len(args),
                    device=device,
                )
                with torch.no_grad():
                    actions, _, _ = actor.get_action(
                        obs_tensor,
                        state_sequence=state_sequence,
                    )

            else:
                with torch.no_grad():
                    actions, _, _ = actor.get_action(obs_tensor)

            actions = actions.detach().cpu().numpy()
        
        # TRY NOT TO MODIFY: execute the game and log data
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if args.model_type == "grumlp" and gru_hidden is not None:
            done_mask = torch.as_tensor(
                np.logical_or(terminations, truncations),
                dtype=torch.bool,
                device=device,
            )
            if done_mask.any():
                gru_hidden[:, done_mask, :] = 0.0

        if args.model_type in ["grumlp", "gnet"]:
            gru_history_cache.append(
                make_gru_history_item(obs[0], actions[0], rewards[0])
            )
            gru_history_cache[:] = gru_history_cache[-int(args.gru_seq_len) :]
            if bool(np.logical_or(terminations, truncations)[0]):
                gru_history_cache.clear()

        if args.model_type in ["state-gru", "state-gru-traj50"]:
            state_gru_cache.append(np.asarray(obs[0], dtype=np.float32).reshape(-1))
            state_gru_cache[:] = state_gru_cache[-max(state_gru_sequence_len(args) - 1, 0) :]
            if bool(np.logical_or(terminations, truncations)[0]):
                state_gru_cache.clear()

        if args.model_type in memory_types + raw_types:
            if args.model_type in hybridrarl_types:
                memory_traj_len = args.hybridrarl_traj_len
            elif args.model_type in hybrid_types:
                memory_traj_len = args.hr_traj_len
            else:
                memory_traj_len = args.ragnet_traj_len

            state_to_cache = np.asarray(obs[0], dtype=np.float32).reshape(-1)
            action_to_cache = np.asarray(actions[0], dtype=np.float32).reshape(-1)
            reward_to_cache = np.asarray([rewards[0]], dtype=np.float32).reshape(-1)

            assert state_to_cache.shape == (obs_dim,), \
                f"cached state shape should be {(obs_dim,)}, got {state_to_cache.shape}"

            assert action_to_cache.shape == (act_dim,), \
                f"cached action shape should be {(act_dim,)}, got {action_to_cache.shape}"

            assert reward_to_cache.shape == (1,), \
                f"cached reward shape should be {(1,)}, got {reward_to_cache.shape}"

            cache_states.append(state_to_cache)
            cache_actions.append(action_to_cache)
            cache_rewards.append(reward_to_cache)

            if len(cache_states) == memory_traj_len:
                traj_states = torch.as_tensor(
                    np.stack(cache_states, axis=0),
                    dtype=torch.float32,
                    device=device,
                )

                traj_actions = torch.as_tensor(
                    np.stack(cache_actions, axis=0),
                    dtype=torch.float32,
                    device=device,
                )

                traj_rewards = torch.as_tensor(
                    np.stack(cache_rewards, axis=0),
                    dtype=torch.float32,
                    device=device,
                )

                assert traj_states.shape == (memory_traj_len, obs_dim), \
                    f"traj_states should be {(memory_traj_len, obs_dim)}, got {traj_states.shape}"

                assert traj_actions.shape == (memory_traj_len, act_dim), \
                    f"traj_actions should be {(memory_traj_len, act_dim)}, got {traj_actions.shape}"

                assert traj_rewards.shape == (memory_traj_len, 1), \
                    f"traj_rewards should be {(memory_traj_len, 1)}, got {traj_rewards.shape}"

                longterm_memory.add_trajectory(
                    states=traj_states,
                    actions=traj_actions,
                    rewards=traj_rewards,
                )

                cache_states.clear()
                cache_actions.clear()
                cache_rewards.clear()

        if args.model_type in rarl_types:
            longterm_memory.add_pair(
                state=np.asarray(obs[0], dtype=np.float32).reshape(-1),
                action=np.asarray(actions[0], dtype=np.float32).reshape(-1),
                reward=np.asarray([rewards[0]], dtype=np.float32).reshape(-1),
            )

        if args.model_type in summary_memory_types:
            state_to_cache = np.asarray(obs[0], dtype=np.float32).reshape(-1)
            reward_to_cache = np.asarray([rewards[0]], dtype=np.float32).reshape(-1)

            assert state_to_cache.shape == (obs_dim,), \
                f"cached state shape should be {(obs_dim,)}, got {state_to_cache.shape}"
            assert reward_to_cache.shape == (1,), \
                f"cached reward shape should be {(1,)}, got {reward_to_cache.shape}"

            cache_states.append(state_to_cache)
            cache_rewards.append(reward_to_cache)

            if len(cache_states) == args.sm_traj_len:
                traj_states = torch.as_tensor(
                    np.stack(cache_states, axis=0),
                    dtype=torch.float32,
                    device=device,
                )
                traj_return = float(np.sum(cache_rewards))

                assert traj_states.shape == (args.sm_traj_len, obs_dim), \
                    f"traj_states should be {(args.sm_traj_len, obs_dim)}, got {traj_states.shape}"

                with torch.no_grad():
                    traj_summary = actor.model.summarize_trajectory(traj_states).squeeze(0)

                write_info = summary_memory.add_summary(
                    summary=traj_summary.detach().cpu(),
                    trajectory_return=traj_return,
                )

                writer.add_scalar(
                    "charts/summary_memory_size",
                    len(summary_memory),
                    global_step,
                )
                writer.add_scalar(
                    "charts/summary_memory_last_return",
                    traj_return,
                    global_step,
                )
                if global_step % 1000 == 0:
                    print(
                        "SummaryMemory write: "
                        f"action={write_info.action}, size={len(summary_memory)}, "
                        f"return={traj_return:.2f}, similarity={write_info.similarity:.3f}"
                    )

                cache_states.clear()
                cache_rewards.clear()

            if bool(np.logical_or(terminations, truncations)[0]):
                cache_states.clear()
                cache_rewards.clear()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}, success={info['success']}"
                )
                writer.add_scalar(
                    'charts/episodic_return', info['episode']['r'], global_step
                )
                writer.add_scalar(
                    'charts/episodic_length', info['episode']['l'], global_step
                )
                writer.add_scalar(
                    'charts/success', info['success'], global_step
                )
                break
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb_dones = (
            np.logical_or(terminations, truncations)
            if args.model_type in ["grumlp", "gnet", "state-gru", "state-gru-traj50"]
            else terminations
        )
        rb.add(obs, real_next_obs, actions, rewards, rb_dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if args.model_type in ["grumlp", "gnet"]:
                data = sample_gru_replay_batch(
                    replay_buffer=rb,
                    batch_size=args.batch_size,
                    history_len=args.gru_seq_len,
                    device=device,
                )
            elif args.model_type in ["state-gru", "state-gru-traj50"]:
                data = sample_state_gru_replay_batch(
                    replay_buffer=rb,
                    batch_size=args.batch_size,
                    sequence_len=state_gru_sequence_len(args),
                    device=device,
                )
            else:
                data = rb.sample(args.batch_size)
            ragnet_kwargs = {}

            if args.model_type in hybridrarl_types + hybrid_types + raw_types:
                retrieve_n_traj = (
                    args.hybridrarl_memory_capacity
                    if args.model_type in hybridrarl_types
                    else args.hr_n_traj
                )
                memory_batch = sample_ragnet_memory_batch(
                    longterm_memory=longterm_memory,
                    n_traj=retrieve_n_traj,
                    device=device,
                )
                ragnet_kwargs["memory_batch"] = memory_batch

            elif args.model_type in ["ragnet", "ragnet-wo-context", "ragnet-wo-a", "ragnet-concat-only"]:
                memory_batch = sample_ragnet_memory_batch(
                    longterm_memory=longterm_memory,
                    n_traj=args.ragnet_n_traj,
                    device=device,
                )
                ragnet_kwargs["memory_batch"] = memory_batch

            with torch.no_grad():
                next_actor_obs = (
                    data.next_observations
                    if args.model_type in ["grumlp", "gnet"]
                    else data.next_observations
                )
                actor_history_kwargs = (
                    {"history": data.next_histories}
                    if args.model_type in ["grumlp", "gnet"]
                    else {}
                )
                actor_state_kwargs = (
                    {"state_sequence": data.next_state_sequences}
                    if args.model_type in ["state-gru", "state-gru-traj50"]
                    else {}
                )
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    next_actor_obs,
                    **actor_history_kwargs,
                    **actor_state_kwargs,
                    **ragnet_kwargs,
                )

                if args.model_type in hybridrarl_types:
                    next_q_obs = data.next_observations
                elif args.model_type in hybrid_types:
                    # Legacy hybrid variants use their encoded state as critic input.
                    next_q_obs = actor.model.encode_state(data.next_observations)
                elif args.model_type == "rarl-reward-ctx":
                    next_z = actor.model.retrieve_context(data.next_observations)
                    next_q_obs = torch.cat([data.next_observations, next_z], dim=-1)
                elif args.model_type == "grumlp":
                    next_q_obs = data.next_observations
                else:
                    next_q_obs = data.next_observations

                q_history_kwargs = (
                    {"history": data.next_histories}
                    if args.model_type == "grumlp"
                    else {}
                )
                qf1_next_target = qf1_target(
                    next_q_obs,
                    next_state_actions,
                    **q_history_kwargs,
                )
                qf2_next_target = qf2_target(
                    next_q_obs,
                    next_state_actions,
                    **q_history_kwargs,
                )
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            if args.model_type in hybridrarl_types:
                q_obs = data.observations
            elif args.model_type in hybrid_types:
                # Legacy hybrid variants use their encoded state as critic input.
                with torch.no_grad():
                    q_obs = actor.model.encode_state(data.observations)
            elif args.model_type == "rarl-reward-ctx":
                with torch.no_grad():
                    z_obs = actor.model.retrieve_context(data.observations)
                    q_obs = torch.cat([data.observations, z_obs], dim=-1)
            elif args.model_type == "grumlp":
                q_obs = data.observations
            else:
                q_obs = data.observations

            q_history_kwargs = (
                {"history": data.histories}
                if args.model_type == "grumlp"
                else {}
            )
            qf1_a_values = qf1(q_obs, data.actions, **q_history_kwargs).view(-1)
            qf2_a_values = qf2(q_obs, data.actions, **q_history_kwargs).view(-1)

            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0: # TD 3 Delayed update support
                for _ in range(args.policy_frequency):
                    # compensate for the delay by doing 'actor_update_interval' instead of 1
                    actor_obs = (
                        data.observations
                        if args.model_type in ["grumlp", "gnet"]
                        else data.observations
                    )
                    actor_history_kwargs = (
                        {"history": data.histories}
                        if args.model_type in ["grumlp", "gnet"]
                        else {}
                    )
                    actor_state_kwargs = (
                        {"state_sequence": data.state_sequences}
                        if args.model_type in ["state-gru", "state-gru-traj50"]
                        else {}
                    )
                    pi, log_pi, _ = actor.get_action(
                        actor_obs,
                        **actor_history_kwargs,
                        **actor_state_kwargs,
                        **ragnet_kwargs,
                    )
                    
                    if args.model_type in hybridrarl_types:
                        q_pi_obs = data.observations
                    elif args.model_type in hybrid_types:
                        # Legacy hybrid variants allow gradient through their encoder.
                        q_pi_obs = actor.model.encode_state(data.observations)
                    elif args.model_type == "rarl-reward-ctx":
                        z_pi_obs = actor.model.retrieve_context(data.observations)
                        q_pi_obs = torch.cat([data.observations, z_pi_obs], dim=-1)
                    elif args.model_type == "grumlp":
                        q_pi_obs = data.observations
                    else:
                        q_pi_obs = data.observations

                    q_pi_history_kwargs = (
                        {"history": data.histories}
                        if args.model_type == "grumlp"
                        else {}
                    )
                    qf1_pi = qf1(q_pi_obs, pi, **q_pi_history_kwargs)
                    qf2_pi = qf2(q_pi_obs, pi, **q_pi_history_kwargs)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    aux_loss = None
                    aux_info = None
                    actor_total_loss = actor_loss

                    actor_optimizer.zero_grad()
                    actor_total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(),
                        max_norm=10.0,
                    )

                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(
                                actor_obs,
                                **actor_history_kwargs,
                                **actor_state_kwargs,
                                **ragnet_kwargs,
                            )
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                soft_update_modules(
                    source_modules=(qf1, qf2),
                    target_modules=(qf1_target, qf2_target),
                    tau=args.tau,
                )
            if global_step % 100 == 0:
                writer.add_scalar(
                    'losses/qf1_values', qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    'losses/qf2_values', qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    'losses/qf1_loss', qf1_loss.item(), global_step
                )
                writer.add_scalar(
                    'losses/qf2_loss', qf2_loss.item(), global_step
                )
                writer.add_scalar(
                    "losses/qf_loss", qf_loss.item() / 2.0, global_step
                )
                writer.add_scalar(
                    'losses/actor_loss', actor_loss.item(), global_step
                )
                writer.add_scalar(
                    'losses/alpha', alpha, global_step
                )
                if args.model_type in hybridrarl_types + hybrid_types + raw_types:
                    if hasattr(actor.model, "net_mean") and hasattr(actor.model.net_mean, "retrieval_gate"):
                        writer.add_scalar(
                            "charts/hybrid_retrieval_gate",
                            torch.sigmoid(actor.model.net_mean.retrieval_gate).item(),
                            global_step,
                        )
                    writer.add_scalar(
                        "charts/longterm_memory_size",
                        len(longterm_memory),
                        global_step,
                    )
                if args.model_type in summary_memory_types:
                    writer.add_scalar(
                        "charts/summary_memory_size",
                        len(summary_memory),
                        global_step,
                    )
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    'charts/SPS',
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

        current_step = global_step + 1
        if args.save_dir is not None and args.save_every > 0 and current_step % args.save_every == 0:
            checkpoint_dir = os.path.join(args.save_dir, run_name)
            print(f"Saving checkpoint at step {current_step} in `{checkpoint_dir}`")
            save_sac_checkpoint(
                dirname=checkpoint_dir,
                actor=actor,
                qf1=qf1,
                qf2=qf2,
                qf1_target=qf1_target,
                qf2_target=qf2_target,
                actor_optimizer=actor_optimizer,
                q_optimizer=q_optimizer,
                args=args,
                global_step=global_step,
                alpha=alpha,
                log_alpha=log_alpha if args.autotune else None,
                a_optimizer=a_optimizer if args.autotune else None,
                save_memory=args.save_memory,
            )

    [
        eval_agent(actor, envs.envs[i], args.num_evals, global_step, writer, device)
        for i in range(envs.num_envs)
    ]

    envs.close()
    writer.close()

    if args.save_dir is not None:
        print(f'Saving trained agent in `{args.save_dir}` with name `{run_name}`')
        save_sac_checkpoint(
            dirname=os.path.join(args.save_dir, run_name),
            actor=actor,
            qf1=qf1,
            qf2=qf2,
            qf1_target=qf1_target,
            qf2_target=qf2_target,
            actor_optimizer=actor_optimizer,
            q_optimizer=q_optimizer,
            args=args,
            global_step=global_step,
            alpha=alpha,
            log_alpha=log_alpha if args.autotune else None,
            a_optimizer=a_optimizer if args.autotune else None,
            save_memory=args.save_memory,
        )
