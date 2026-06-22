import torch
import torch.nn as nn
import os

try:
    from .shared_arch import shared
except ImportError:
    from shared_arch import shared


class GRUMLPAgent(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=256,
        gru_layers=1,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers

        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

        # GRU 的最后隐状态维度是 hidden_dim
        # 所以 shared 的输入维度应该是 hidden_dim
        self.fc = shared(input_dim=hidden_dim)

        self.fc_mean = None
        self.fc_logstd = None
        self.reset_heads()

    def reset_heads(self):
        # 假设 shared(...) 的输出维度是 256
        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

    def forward(self, x, h=None, return_hidden=False):
        """
        x:
            [batch_size, obs_dim]
            or
            [batch_size, seq_len, obs_dim]

        h:
            [gru_layers, batch_size, hidden_dim]

        return:
            默认返回:
                mean, log_std

            如果 return_hidden=True:
                mean, log_std, h
        """

        # 单步输入 [B, obs_dim] -> [B, 1, obs_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # gru_out: [B, seq_len, hidden_dim]
        # h: [gru_layers, B, hidden_dim]
        gru_out, h = self.gru(x, h)

        # 取最后一层 GRU 的最后隐状态
        # shape: [B, hidden_dim]
        last_hidden = h[-1]

        # 把 GRU 最后的 hidden state 送入 MLP
        x = self.fc(last_hidden)

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        if return_hidden:
            return mean, log_std, h

        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/model.pt")

    @staticmethod
    def load(dirname, map_location=None, reset_heads=False):
        model = torch.load(f"{dirname}/model.pt", map_location=map_location)
        if reset_heads:
            model.reset_heads()
        return model


if __name__ == "__main__":
    agent = GRUMLPAgent(obs_dim=4, act_dim=4)

    # 单步输入
    x = torch.randn(8, 4)
    mean, log_std, h = agent(x, return_hidden=True)
    print(mean.shape)      # torch.Size([8, 4])
    print(log_std.shape)   # torch.Size([8, 4])
    print(h.shape)         # torch.Size([1, 8, 256])

    # 序列输入
    x_seq = torch.randn(8, 10, 4)
    mean, log_std, h = agent(x_seq, return_hidden=True)
    print(mean.shape)      # torch.Size([8, 4])
    print(log_std.shape)   # torch.Size([8, 4])
    print(h.shape)         # torch.Size([1, 8, 256])

    dirname = "runs/test_gru_mlp"
    agent.save(dirname)