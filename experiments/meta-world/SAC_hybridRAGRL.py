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
from models import SingleNetAgent, shared, SimpleAgent, GRUMLPAgent, LongTermMemory, HybridSoftQNetwork, HybridRAGRLAgent, GRUStateEncoder
# from models import RARLSACCritic, RARLSACAgent, LongTermMemory
import torch.nn.functional as F


@dataclass
class Args:
    model_type: Literal['simple', 'finetune', 'packnet', 'prognet', 'sdnet', 'grumlp', 'only-sdnet', 'ragnet', 'ragnet-wo-context', 'ragnet-wo-a', 'ragnet-concat-only', 'hybridrarl', 'hybridrarl-wo-gate', 'hybrid_add_norm', 'rarl', 'hybrid-encoder', 'hybrid-context', 'hybrid-raw-encoder', 'hybrid-raw-context', 'hybrid-raw', 'hybrid', 'gate'] 
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
        mean, log_std = self.model(x, **kwargs)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) # From SpinUp / Dennis Yarats

        return mean, log_std
    
    def get_action(self, x, **kwargs):
        mean, log_std = self(x, **kwargs)
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
        return action, log_prob, mean

@torch.no_grad()
def eval_agent(agent, test_env, num_evals, global_step, writer, device):
    obs, _ = test_env.reset()
    avg_ep_ret = 0
    avg_success = 0
    ep_ret = 0
    for _ in range(num_evals):
        while True:
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            action, _ = agent(obs)
            obs, reward, termination, truncation, info = test_env.step(
                action[0].cpu().numpy()
            )

            ep_ret += reward

            if termination or truncation:
                print(f'info is \n{info}')
                avg_success += info['success']
                avg_ep_ret += ep_ret
                #reset
                obs, _ = test_env.reset()
                ep_ret = 0
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
    if save_memory and hasattr(model, "memory"):
        model.save(dirname=dirname, save_memory=True)
    else:
        model.save(dirname=dirname)


def longterm_capacity_from_transition_budget(buffer_size, traj_len):
    return max(1, int(buffer_size) // int(traj_len))

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

    hybrid_types = [
        "hybridrarl",
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

    rarl_types = [
        "rarl",
    ]

    memory_types = ragnet_types + hybrid_types + rarl_types
    hr_memory_types = hybrid_types + rarl_types
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
        model = GRUMLPAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)
    elif args.model_type == 'finetune':
        model = SimpleAgent.load(
            args.prev_units[0], map_location=device, reset_heads=True
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
    
    elif args.model_type == "rarl":
        longterm_capacity = longterm_capacity_from_transition_budget(
            args.buffer_size,
            args.traj_len,
        )
        print(
            f"*** LongTermMemory capacity: {longterm_capacity} trajectories "
            f"({longterm_capacity * args.traj_len} transitions)"
        )

        longterm_memory = LongTermMemory(
            capacity=longterm_capacity,
            traj_len=args.traj_len,
            obs_dim=obs_dim,
            action_dim=act_dim,
            reward_dim=1,
            device=device,
        )

        model = RARLSACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            memory=longterm_memory,
            n_traj=args.hr_n_traj,
            state_dim=128,
            summary_dim=128,
            slot_dim=128,
            z_dim=128,
            num_slots=4,
            top_k=8,
        ).to(device)

        actor = Actor(envs, model).to(device)

    elif args.model_type in ['sdnet', 'sdnet-rapid', 'only-sdnet']:
        model = SingleNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            buffer=rb,
            map_location=device,
        ).to(device)
    # Ignore the ProgNet and PackNet methods

    actor = Actor(envs, model).to(device)
    if args.model_type in hybrid_types:
        critic_feature_dim = int(args.hr_state_dim)
    elif args.model_type in ["ragnet", "ragnet-wo-context", "ragnet-wo-a", "ragnet-concat-only"]:
        critic_feature_dim = actor.model.net_mean.policy_input_dim,
    else:
        critic_feature_dim = obs_dim

    # critic_feature_dim = critic_feature_dim[0]
    # obs_dim = np.array(envs.single_observation_space.shape).prod()
    # act_dim = np.prod(envs.single_action_space.shape)
    print("critic_feature_dim =", critic_feature_dim, type(critic_feature_dim))
    print("act_dim =", act_dim, type(act_dim))
    print("critic input dim =", critic_feature_dim + act_dim, type(critic_feature_dim + act_dim))

    if args.model_type == "rarl":
        qf1 = RARLSACCritic(state_dim=128, act_dim=act_dim).to(device)
        qf2 = RARLSACCritic(state_dim=128, act_dim=act_dim).to(device)
        qf1_target = RARLSACCritic(state_dim=128, act_dim=act_dim).to(device)
        qf2_target = RARLSACCritic(state_dim=128, act_dim=act_dim).to(device)
    
    else:
        qf1 = SoftQNetwork(critic_feature_dim, act_dim).to(device)
        qf2 = SoftQNetwork(critic_feature_dim, act_dim).to(device)
        qf1_target = SoftQNetwork(critic_feature_dim, act_dim).to(device)
        qf2_target = SoftQNetwork(critic_feature_dim, act_dim).to(device)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
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
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        
        # ALGO LOGIC: put action logic here
        if global_step < args.random_actions_end:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            obs_tensor = torch.Tensor(obs).to(device)

            if args.model_type in hybrid_types + raw_types:
                assert len(longterm_memory) > 0, (
                    "Hybrid RAGRL starts acting after random_actions_end, but LongTermMemory is empty. "
                    "Increase random_actions_end or decrease hr_traj_len."
                )

                memory_batch = sample_ragnet_memory_batch(
                    longterm_memory=longterm_memory,
                    n_traj=args.hr_n_traj,
                    device=device,
                )

                actions, _, _ = actor.get_action(
                    obs_tensor,
                    memory_batch=memory_batch,
                )
            
            elif args.model_type == "rarl":
                assert len(longterm_memory) > 0, (
                    "RARL starts acting after random_actions_end, but LongTermMemory is empty. "
                    "Increase random_actions_end or decrease hr_traj_len."
                )

                memory_batch = sample_ragnet_memory_batch(
                    longterm_memory=longterm_memory,
                    n_traj=args.hr_n_traj,
                    device=device,
                )

                actions, _, _ = actor.get_action(
                    obs_tensor,
                    memory_batch=memory_batch,
                    deterministic_z=True,
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

            else:
                actions, _, _ = actor.get_action(obs_tensor)

            actions = actions.detach().cpu().numpy()
        
        # TRY NOT TO MODIFY: execute the game and log data
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if args.model_type in memory_types + raw_types:
            memory_traj_len = args.hr_traj_len if args.model_type in hybrid_types else args.ragnet_traj_len

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
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            ragnet_kwargs = {}

            if args.model_type in hybrid_types + raw_types:
                memory_batch = sample_ragnet_memory_batch(
                    longterm_memory=longterm_memory,
                    n_traj=args.hr_n_traj,
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
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations,
                    **ragnet_kwargs,
                )

                if args.model_type in hybrid_types:
                    # Hybrid critic input is encoded state s_{t+1}.
                    next_q_obs = actor.model.encode_state(data.next_observations)
                else:
                    next_q_obs = data.next_observations

                qf1_next_target = qf1_target(next_q_obs, next_state_actions)
                qf2_next_target = qf2_target(next_q_obs, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            # with torch.no_grad():
            #     next_actions, next_log_pi, _ = actor.get_action(
            #         data.next_observations,
            #         memory_batch=memory_batch,
            #         deterministic_z=True,
            #     )

            #     next_s_tilde = actor.model.compute_features(
            #         data.next_observations,
            #         memory_batch=memory_batch,
            #         deterministic_z=True,
            #         return_info=False,
            #     )

            #     qf1_next_target = qf1_target(next_s_tilde, next_actions)
            #     qf2_next_target = qf2_target(next_s_tilde, next_actions)

            #     min_qf_next_target = (
            #         torch.min(qf1_next_target, qf2_next_target)
            #         - alpha * next_log_pi
            #     )

            #     next_q_value = data.rewards.flatten() + (
            #         1 - data.dones.flatten()
            #     ) * args.gamma * min_qf_next_target.view(-1)


            if args.model_type in hybrid_types:
                # Detach encoded state during critic update.
                with torch.no_grad():
                    q_obs = actor.model.encode_state(data.observations)
            elif args.model_type == "rarl":
                with torch.no_grad():
                    q_obs = actor.model.compute_features(
                        data.observations,
                        memory_batch=memory_batch,
                        deterministic_z=True,
                        return_info=False,
                    )
            else:
                q_obs = data.observations

            qf1_a_values = qf1(q_obs, data.actions).view(-1)
            qf2_a_values = qf2(q_obs, data.actions).view(-1)

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
                    pi, log_pi, _ = actor.get_action(
                        data.observations,
                        **ragnet_kwargs,
                    )
                    
                    if args.model_type in hybrid_types:
                        # Actor update should allow gradient to flow through encoder.
                        q_pi_obs = actor.model.encode_state(data.observations)
                    elif args.model_type == "rarl":
                        with torch.no_grad():
                            q_pi_obs = actor.model.compute_features(
                                data.observations,
                                memory_batch=memory_batch,
                                deterministic_z=True,
                                return_info=False,
                            )
                    else:
                        q_pi_obs = data.observations

                    qf1_pi = qf1(q_pi_obs, pi)
                    qf2_pi = qf2(q_pi_obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    if args.model_type == "rarl":
                        aux_loss, aux_info = actor.model.compute_aux_loss(
                            memory_batch=memory_batch,
                            gamma=args.gamma,
                            reward_coef=args.rarl_aux_reward_coef,
                            return_coef=args.rarl_aux_return_coef,
                            action_coef=args.rarl_aux_action_coef,
                            normalize_targets=True,
                        )

                        actor_total_loss = actor_loss + args.rarl_aux_coef * aux_loss

                    else:
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
                                data.observations,
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
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
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
                if args.model_type in hybrid_types + raw_types:
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
            save_agent(
                actor.model,
                dirname=checkpoint_dir,
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
        save_agent(
            actor.model,
            dirname=os.path.join(args.save_dir, run_name),
            save_memory=args.save_memory,
        )
