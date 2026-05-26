import torch 
import torch.nn as nn
import numpy as np

from .shared_arch import shared
import sys, os

sys.path.append(os.path.dirname(__file__) + "/../../../")
from sdnet import SingleNet

# import wandb


def net(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, output_dim)
    )

def take_first(x):
    return x[0]

class SingleNetAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, buffer, map_location=None):
        super().__init__()

        self.net_logstd = net(obs_dim, act_dim)

        self.net_mean = SingleNet(
            input_dim=obs_dim,
            hidden_dim=256,
            output_dim=act_dim,
            buffer=buffer,
            internal_policy=net(obs_dim + 256, act_dim),
        )
    
    def forward(self, x, writer=None, global_step=None):
        if writer is None or global_step is None:
            mean = self.net_mean(x)[0]
        else:
            mean, att_out, policy_out, lin_val = self.net_mean(x, ret_att_out=True, ret_int_pol=True, ret_lin_val=True)
            for i, v in enumerate(att_out.mean(0)[0].detach()):
                writer.add_scalar(f'charts/att_out_{i}', v.item(), global_step)
            with torch.no_grad():
                dist_out_int_pol = (mean - policy_out).abs().mean().item()
                dist_out_lin_val = (mean - lin_val).abs().mean().item()

                writer.add_scalar(
                    f'charts/dist_out_int_pol', dist_out_int_pol, global_step
                )
                writer.add_scalar(
                    f'charts/dist_out_lin_val', dist_out_lin_val, global_step
                )

        log_std = self.net_logstd(x)
        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.net_mean, f'{dirname}/net_mean.pt')
        torch.save(self.net_logstd, f'{dirname}/net_logstd.pt')

    def load(dirname, obs_dim, act_dim, map_location=None):
        model = SingleNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            map_location=map_location,
        )
        model.net_logstd = torch.load(f'{dirname}/net_logstd.pt', map_location=map_location)

        net_mean = torch.load(f'{dirname}/net_mean.pt', map_location=map_location)

        curr = model.net_mean.state_dict()
        other = net_mean.state_dict()
        for k in other:
            curr[k] = other[k]
        model.net_mean.load_state_dict(curr)

        return model
        
if __name__ == '__main__':
    singlenet = SingleNetAgent(obs_dim=3, act_dim=4)
    print(f'*** logstd net\n{singlenet.net_mean}')
