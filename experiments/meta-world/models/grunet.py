import torch
import torch.nn as nn
import os


class GRUAgent(nn.Module):
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

        self.fc_mean = None
        self.fc_logstd = None
        self.reset_heads()

    def reset_heads(self):
        self.fc_mean = nn.Linear(self.hidden_dim, self.act_dim)
        self.fc_logstd = nn.Linear(self.hidden_dim, self.act_dim)

    def forward(self, x, h=None):
        """
        x:
            [batch_size, obs_dim]
            or
            [batch_size, seq_len, obs_dim]

        h:
            [gru_layers, batch_size, hidden_dim]
        """

        # 如果输入是单步状态 [B, obs_dim]，转成 [B, 1, obs_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        gru_out, h = self.gru(x, h)

        # 取最后一个时间步
        last_out = gru_out[:, -1, :]

        mean = self.fc_mean(last_out)
        log_std = self.fc_logstd(last_out)

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
    agent = GRUAgent(obs_dim=4, act_dim=4)

    # 单步输入
    x = torch.randn(8, 4)
    mean, log_std, h = agent(x)
    print(mean.shape, log_std.shape, h.shape)

    # 序列输入
    x_seq = torch.randn(8, 10, 4)
    mean, log_std, h = agent(x_seq)
    print(mean.shape, log_std.shape, h.shape)

    dirname = "runs/test_gru"
    agent.save(dirname)