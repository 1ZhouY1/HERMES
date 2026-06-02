import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions.categorical import Categorical
from stable_baselines3.common.buffers import ReplayBuffer
# import wandb


def get_position_encoding(seq_len, d, n=10_000):
    """
    Returns the Positional Encoding matrix P for the given parameters.
    
    seq_len -- the length of sequences.
    d       -- the dimension of the model.
    n.      -- User defined scalar, set to 10k in "Attention is all you need". Default: 10_000.
    """
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in range(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i + 1] = np.cos(k/denominator)
    return P
# print(get_position_encoding(3, 5))

buffer = []

def logit2prob(logits):
    return F.softmax(logits, dim=1)



class Identity:
    def __call__(self, arg, **kwargs):
        return arg



class SingleNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        internal_policy: nn.Module,
        encoder: nn.Module = None, 
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        att_head_init: object = Identity(),
        proj_bias: bool = True, 
        buffer: ReplayBuffer = None,
    ):
        """
        SingleNet initialize

        input_dim       -- the dimension of input state
        hidden_dim      -- the dimension of the model
        output_dim      -- the dimension of the action
        encoder         -- the encoder of the state if not None
        internal_policy -- the internal policy
        buffer          -- the replay buffer storing the experiences
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.buffer = buffer
        self.encoder = encoder if encoder is not None else Identity()
        self.internal_policy = internal_policy
        self.head_wq = att_head_init(
            nn.Linear(input_dim, hidden_dim, bias=proj_bias)
            )
        self.head_wk = att_head_init(
            nn.Linear(input_dim, hidden_dim, bias=proj_bias)
        )
        self.head_wv = att_head_init(
            nn.Linear(output_dim, hidden_dim, bias=proj_bias)
        )
        self.att_temp = np.sqrt(
            hidden_dim
        )
        self.top_k = 10
        # len = len(buffer)
        # self.pe = get_position_encoding(len, d=output_dim)

    def sampling(self, k_samples):
        """
        Returns the retrieved samples of the experience buffer
        """
        samples = self.buffer.sample(k_samples)
        return samples
    
    def retrieve(self, s, phi_s, phi_a):
        """
        Returns the most similar state-action pairs and the linear composition of actions.

        s     -- the current state if the encoder is None else the encoded state.
        phi_s -- the retrieved state samples matrix of the experiences from the replay buffer.
        phi_a -- the retrieved action samples matrix of the experiences from the replay buffer.
        """
        query = self.head_wq(s)
        keys = self.head_wk(phi_s)
        values = phi_a
        #W is Q * K^T
        W = torch.matmul(
            # [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
            query[:, None, :],
            # [batch_size, k, hidden_dim] -> [batch_size, hidden_dim, k]
            keys.permute(0, 2, 1) #transpose the keys matrix
        )
        logits = W.squeeze(1) / self.att_temp
        topk_logits, topk_indices = torch.topk(
            logits, 
            self.top_k,
            dim=-1
        )
        weights = F.softmax(topk_logits, dim=-1) # [batch_size, top_k]

        topk_values = torch.gather(
            values,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, values.size(-1))
        )

        retrieved_action = torch.sum(weights.unsqueeze(-1) * topk_values, dim=1)

        return retrieved_action, weights
        
    
    def attention_out(self, s, phi_s, phi_a):
        """
        Returns the most similar state-action pairs and the linear composition of actions.

        s     -- the input state if the encoder is None.
        phi_s -- the retrieved state samples matrix of the experiences from the replay buffer.
        phi_a -- the retrieved action samples matrix of the experiences from the replay buffer.
        """
        query = self.head_wq(s)
        keys = self.head_wk(phi_s)
        values = self.head_wv(phi_a)
        #W is Q * K^T
        W = torch.matmul(
            #size: [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
            query[:, None, :],
            #size: [batch_size, k, hidden_dim] -> [batch_size, hidden_dim, k]
            keys.permute(0, 2, 1) #transpose the keys matrix
        )
        weight = F.softmax(W / self.att_temp, dim=-1)

        a_att = torch.matmul(weight, values)
        a_att = a_att[:, 0, :] #remove extra dim

        return a_att, weight
    
    def policy_out(self, s):
        """
        Returns the policy output. Because of Soft Actor-Critic, the output of policy might be the mean and std. By using 
        reparameterization trick we can get the sample of actions.

        s -- the observation of the current state.
        """

        # the original version
        policy_out = self.internal_policy(s)

        return policy_out
    
    def forward(
            self, 
            s, 
            k_samples=128,
            ret_att_out=False, 
            ret_int_pol=False, 
            ret_lin_val=False, 
            buffer4test=None
        ):
        """
        Returns the final output action.
        
        s               --  the observation of the current state.

        k_samples       --  the number of samples.

        ret_att_out     --  If true, the output of the attention are returned.
                            Default: False
        
        ret_int_pol     --  If true, the output of the internal policy without normalization will be returned.
                            Default: False
        
        ret_lin_val     --  If true, the output of the linear composition of values will be returned.
                            Default: False

        buffer4test     --  define a buffer or matrix for test this method.
                            Default: None

        """

        batch_size = s.shape[0]
        # Obtain the concat matrix of the samples from the replay buffer
        if buffer4test is not None:
            # To test this method
            att_out, att_weight = self.attention_out(s, phi_s=buffer4test[0], phi_a=buffer4test[1])
            lin_val = torch.matmul(att_weight, buffer4test[1])

        elif self.buffer is not None and self.buffer.size() >= k_samples:
            samples = self.buffer.sample(k_samples)

            phi_s_base = samples.observations
            phi_a_base = samples.actions

            phi_s = phi_s_base.unsqueeze(0).expand(batch_size, -1, -1)
            phi_a = phi_a_base.unsqueeze(0).expand(batch_size, -1, -1)

            # rapid_out, rapid_weight = self.rapid(s, phi_s_topk, phi_a_topk)

            att_out, att_weight = self.retrieve(s, phi_s, phi_a)

        # concat context and current state
        # policy_in = torch.hstack([att_out, s])
        # policy_out = self.policy_out(policy_in)

        policy_out = self.policy_out(s)
        # out_temp = policy_out + att_out
        # a_temp = att_out + self.policy_out

        # Normalization
        # final_out = logit2prob(out_temp)

        ret_vals = [policy_out]
        if ret_att_out:
            ret_vals.append(att_weight)
        if ret_int_pol:
            ret_vals.append(policy_out)
        if ret_lin_val:
            ret_vals.append(att_out)
        return ret_vals
    
if __name__ == "__main__":
    import gymnasium as gym
    import random
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # total_steps = 1000
    # random_steps = 200

    # in_dim = env.observation_space.shape[0]
    # hidden_dim = 512
    # out_dim = env.action_space.shape[0]

    in_dim = 3
    hidden_dim = 256
    out_dim = 2

    internal_policy = nn.Sequential(
        nn.Linear(in_dim + hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ).to(device)


    # buffer = ReplayBuffer(
    #     buffer_size=500,
    #     observation_space=env.observation_space,
    #     action_space = env.action_space,
    #     device='cuda',
    #     n_envs=1,
    # ).to(device)


    singlenet = SingleNet(
        input_dim=in_dim, 
        hidden_dim=hidden_dim,
        output_dim=out_dim,
        internal_policy=internal_policy,
        device='cuda',
        buffer=None,
        ).to(device)
    
    # obs, _ = env.reset(seed=1)
    # start_time = time.time()

    query_init = torch.tensor([[1,1,1]], dtype=torch.float32).to(device)
    key_init = torch.tensor([[[1,1,1], [2,2,2]]], dtype=torch.float32).to(device)
    value_init = torch.tensor([[[0.1, 0.2], [0.2, 0.1]]], dtype=torch.float32).to(device)

    # att_out, _ = singlenet.attention_out(query_init, key_init, value_init)
    # print(f'output of attention head is \n{att_out}')
    # print(f'the shape of output is {att_out.shape}')
    buffer4test = [key_init, value_init]
    # print(buffer4test)

    out, att_out, policy_out, lin_val = singlenet(query_init, ret_att_out=True, ret_int_pol=True, ret_lin_val=True, k_samples=None, buffer4test=buffer4test)
    print(f'the final output is {out}')
    print(f'the shape of output is {out.shape}')
    print(f'the output of attention head is {att_out}')
    print(f'the mean:{att_out.mean(0)[0]}')
    print(f'the shape of this is {att_out.shape}')
    print(f'the output of policy is {policy_out}')
    print(f'the shape of this is {policy_out.shape}')
    print(f'the linear composition of values is {lin_val}')
    print(f'the shape of this is {lin_val.shape}')

    # for i in range(total_steps):
    #     if i <= random_steps:
    #         action = np.array(
    #             [envs.single_action_space.sample() for _ in range(envs.num_envs)]
    #         )
    #     elif:
    #         action = singlenet.forward(obs)
    #         action = action.detach().cpu().numpy()
    #     next_obs, reward, termination, truncation, infos = env.step(action)

            
