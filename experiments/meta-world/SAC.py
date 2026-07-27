import torch
import torch.nn as nn
import numpy as np
from tasks import get_task
import gymnasium as gym
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import pathlib
import os
import tyro
from torch.utils.tensorboard import SummaryWriter
import random
import time
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from models import SingleNetAgent, shared, SimpleAgent, GRUMLPAgent, RecurrentSoftQNetwork, sample_gru_replay_batch, LongTermMemory, HybridSoftQNetwork, HybridRAGRLAgent, GRUStateEncoder
import torch.nn.functional as F


@dataclass
class Args:
    model_type: Literal['simple', 'finetune', 'packnet', 'prognet', 'sdnet', 'grumlp', 'only-sdnet', 'ragnet', 'ragnet-wo-context', 'ragnet-wo-a', 'ragnet-concat-only', 'hybridrarl'] 
    """The name of the NN model to use for the agent"""
    save_dir: Optional[str] = None
    """If provided, the model will be saved in the given directory"""
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

    # for hybrid RARL
    hr_n_traj = 512
    hr_traj_len: int = 50



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
        self.fc = shared(
            feature_dim + act_dim
        ) 
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
    for _ in range(num_evals):
        while True:
            obs_np = obs
            obs = torch.Tensor(obs_np).to(device).unsqueeze(0)
            if isinstance(agent.model, GRUMLPAgent):
                gru_history = build_gru_history_sequence(
                    history_cache=gru_cache,
                    obs_dim=agent.model.obs_dim,
                    act_dim=agent.model.act_dim,
                    history_len=getattr(agent.model, "context_seq_len", 20),
                    device=device,
                )
                _, _, action = agent.get_action(obs, history=gru_history)
            else:
                _, _, action = agent.get_action(obs)
            action_np = action[0].cpu().numpy()
            obs, reward, termination, truncation, info = test_env.step(action_np)

            if isinstance(agent.model, GRUMLPAgent):
                gru_cache.append(make_gru_history_item(obs_np, action_np, reward))
                gru_cache[:] = gru_cache[-int(getattr(agent.model, "context_seq_len", 20)) :]

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
                break
    avg_ep_ret /= num_evals
    avg_success /= num_evals
    print(f'\nTEST: ep_ret={avg_ep_ret}, success={avg_success}\n')
    writer.add_scalar("charts/test_episodic_return", avg_ep_ret, global_step)
    writer.add_scalar("charts/test_success", avg_success, global_step)

def sample_ragnet_memory_batch(longterm_memory, n_traj, device):
    traj_batch = longterm_memory.sample_trajectories(
        n_traj=n_traj,
        device=device,
    )

    return {
        "obs": traj_batch.states,
        "actions": traj_batch.actions,
        "rewards": traj_batch.rewards,
    }


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
):
    os.makedirs(dirname, exist_ok=True)
    actor.model.save(dirname=dirname)

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


if __name__ == '__main__':
    # envs = gym.vector.SyncVectorEnv([make_env(0)]) # The SyncVectorEnv method requires a list of function as input
    # high = envs.single_action_space.high
    # low = envs.single_action_space.low
    # print(f'high: {high}') # [1,1,1,1]
    # print(f'low: {low}') # [-1,-1,-1,-1]

    args = tyro.cli(Args)

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

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.task_id)])
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    #select the model to use as the agent
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
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
    cache_states = []
    cache_actions = []
    cache_rewards = []

    if args.model_type == 'finetune':
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
    elif args.model_type == 'finetune':
        model = SimpleAgent.load(
            args.prev_units[0], map_location=device, reset_heads=True
        ).to(device)
    elif args.model_type in ['ragnet', 'ragnet-wo-context', 'ragnet-wo-a', 'ragnet-concat-only']:

        longterm_memory = LongTermMemory(
            capacity=args.buffer_size,
            traj_len=50,
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
    elif args.model_type in ['sdnet', 'sdnet-rapid', 'only-sdnet']:
        model = SingleNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            buffer=rb,
            map_location=device,
        ).to(device)
    # Ignore the ProgNet and PackNet methods

    actor = Actor(envs, model).to(device)
    
    if args.model_type in ["ragnet", "ragnet-wo-context", "ragnet-wo-a", "ragnet-concat-only"]:
        critic_feature_dim = actor.model.net_mean.policy_input_dim
    else:
        critic_feature_dim = obs_dim

    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
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

            if args.model_type in ["ragnet", "ragnet-wo-context", 'ragnet-wo-a', 'ragnet-concat-only']:
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
                    actions, _, _ = actor.get_action(
                        obs_tensor,
                        memory_batch=memory_batch,
                        writer=writer,
                        global_step=global_step,
                    )
                else:
                    actions, _, _ = actor.get_action(
                        obs_tensor,
                        memory_batch=memory_batch,
                    )

            elif args.model_type in ["component", 'sdnet', 'sdnet-rapid'] and global_step % 1000 == 0:
                actions, _, _ = actor.get_action(
                    obs_tensor,
                    writer=writer,
                    global_step=global_step,
                )

            elif args.model_type == "grumlp":
                gru_history = build_gru_history_sequence(
                    history_cache=gru_history_cache,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    history_len=args.gru_seq_len,
                    device=device,
                )
                with torch.no_grad():
                    actions, _, _ = actor.get_action(obs_tensor, history=gru_history)

            else:
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

        if args.model_type == "grumlp":
            gru_history_cache.append(
                make_gru_history_item(obs[0], actions[0], rewards[0])
            )
            gru_history_cache[:] = gru_history_cache[-int(args.gru_seq_len) :]
            if bool(np.logical_or(terminations, truncations)[0]):
                gru_history_cache.clear()

        if args.model_type in ["ragnet", "ragnet-wo-context", "ragnet-wo-a", "ragnet-concat-only"]:
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

            if len(cache_states) == 50:
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

                assert traj_states.shape == (args.ragnet_traj_len, obs_dim), \
                    f"traj_states should be {(args.ragnet_traj_len, obs_dim)}, got {traj_states.shape}"

                assert traj_actions.shape == (args.ragnet_traj_len, act_dim), \
                    f"traj_actions should be {(args.ragnet_traj_len, act_dim)}, got {traj_actions.shape}"

                assert traj_rewards.shape == (args.ragnet_traj_len, 1), \
                    f"traj_rewards should be {(args.ragnet_traj_len, 1)}, got {traj_rewards.shape}"

                longterm_memory.add_trajectory(
                    states=traj_states,
                    actions=traj_actions,
                    rewards=traj_rewards,
                )

                cache_states.clear()
                cache_actions.clear()
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
            if args.model_type == "grumlp"
            else terminations
        )
        rb.add(obs, real_next_obs, actions, rewards, rb_dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if args.model_type == "grumlp":
                data = sample_gru_replay_batch(
                    replay_buffer=rb,
                    batch_size=args.batch_size,
                    history_len=args.gru_seq_len,
                    device=device,
                )
            else:
                data = rb.sample(args.batch_size)
            ragnet_kwargs = {}

            if args.model_type in ["ragnet", "ragnet-wo-context", "ragnet-wo-a", "ragnet-concat-only"]:
                memory_batch = sample_ragnet_memory_batch(
                    longterm_memory=longterm_memory,
                    n_traj=args.ragnet_n_traj,
                    device=device,
                )
                ragnet_kwargs["memory_batch"] = memory_batch
            with torch.no_grad():
                next_actor_obs = (
                    data.next_observations
                    if args.model_type == "grumlp"
                    else data.next_observations
                )
                actor_history_kwargs = (
                    {"history": data.next_histories}
                    if args.model_type == "grumlp"
                    else {}
                )
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    next_actor_obs,
                    **actor_history_kwargs,
                    **ragnet_kwargs,
                )
                next_q_obs = (
                    data.next_observations
                    if args.model_type == "grumlp"
                    else data.next_observations
                )
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

            q_obs = (
                data.observations
                if args.model_type == "grumlp"
                else data.observations
            )
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
                        if args.model_type == "grumlp"
                        else data.observations
                    )
                    actor_history_kwargs = (
                        {"history": data.histories}
                        if args.model_type == "grumlp"
                        else {}
                    )
                    pi, log_pi, _ = actor.get_action(
                        actor_obs,
                        **actor_history_kwargs,
                        **ragnet_kwargs,
                    )
                    
                    q_pi_obs = (
                        data.observations
                        if args.model_type == "grumlp"
                        else data.observations
                    )
                    q_pi_history_kwargs = (
                        {"history": data.histories}
                        if args.model_type == "grumlp"
                        else {}
                    )
                    qf1_pi = qf1(q_pi_obs, pi, **q_pi_history_kwargs)
                    qf2_pi = qf2(q_pi_obs, pi, **q_pi_history_kwargs)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()

                    # if args.model_type == 'packnet':
                    #     if global_step >= packnet_retain_start:
                    #         # can be called multiple times, only the first counts
                    #         actor.model.start_retaining()
                    #     actor.model.before_update()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(
                                actor_obs,
                                **actor_history_kwargs,
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

    [
        eval_agent(actor, envs.envs[i], args.num_evals, global_step, writer, device)
        for i in range(envs.num_envs)
    ]

    envs.close()
    writer.close()

    if args.save_dir is not None:
        print(f'Saving trained agent in `{args.save_dir}` with name `{run_name}`')
        save_sac_checkpoint(
            dirname=f'{args.save_dir}/{run_name}',
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
        )
