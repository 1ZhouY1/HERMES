import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def details_display(info: torch.tensor):
    print(f'{info}')
    print(f'the shape of it is {info.shape}')

Q = torch.tensor([[1,1,1]]) #size: [batch_size, hidden_dim]
details_display(Q)
Q = Q[:, None, :] #size: [batch_size, 1, hidden_dim]
details_display(Q)

K = torch.tensor([[[3,3,3], [4,4,4]]]) #size: [batch_size, k, hidden_dim]
details_display(K)

V = torch.tensor([[[1,2,1,2], [2,1,2,1]]], dtype=torch.float32) #size: [batch_size, k, value_dim]

K = K.permute(0,2,1)
print(f'After permuting:')
details_display(K)

print(f'Now computing Q * K^T')
QKT = torch.matmul(Q, K)
details_display(QKT)

temp = np.sqrt(3)

weights = F.softmax(QKT / temp, dim=-1)
details_display(weights)

attention_output = torch.matmul(weights, V)
details_display(attention_output)

# y_t = torch.tensor([2])
# print(f'tensor.pow function {y_t.pow(2)}')

class Actor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer(
            "action_scale",
            torch.tensor(
                1,
                dtype=torch.float32,
            )
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                0,
                dtype=torch.float32
            )
        )
    def get_out(self, mean, std):
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # x_t is a sample from N(mean, std) by using reparameterization trick
        y_t = x_t
        action = y_t * self.action_scale + self.action_bias
        return action
    
if __name__ == "__main__":
    # actor = Actor()
    # action = actor.get_out(1, 2)
    # print(action)
    # a = torch.tensor([[1,2,3]])
    # print(a.unsqueeze(0).shape)
    # print(a.squeeze(1).shape)
    action = torch.tensor([[1,2,3]])
    print(action.shape)
    target_entropy = -torch.prod(torch.Tensor(action.shape))
    print(target_entropy)