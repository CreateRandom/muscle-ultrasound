import torch
from torch import nn
from torch.nn import functional as F

class AverageMIL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x, dim=0)
        x = x.unsqueeze(0)
        return x, None

class MaxMIL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, _ = torch.max(x, dim=0)
        x = x.unsqueeze(0)
        return x, None


class AttentionMIL(nn.Module):
    def __init__(self, L, D, K, mode='identity'):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K
        self.in_layer = nn.Linear(self.L, self.D)
        self.tanh = nn.Tanh()
        self.out_layer = nn.Linear(self.D, self.K)
        self.mode = mode

    def forward(self, x):
        a = self.in_layer(x)
        a = self.tanh(a)
        a = self.out_layer(a)
        a = torch.transpose(a, 1, 0)
        if self.mode == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.mode == 'sigmoid':
            a = torch.sigmoid(a)

      #  print(a.detach().cpu().numpy())
        x = torch.mm(a, x)
        return x, a

class GatedAttentionMIL(nn.Module):
    def __init__(self, L, D, K, mode='identity'):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.mode = mode


    def forward(self, x):
        a_v = self.attention_V(x)  # NxD
        a_u = self.attention_U(x)  #
        a = self.attention_weights(a_v * a_u)  # element wise multiplication # NxK
        a = torch.transpose(a, 1, 0)  # KxN
        if self.mode == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.mode == 'sigmoid':
            a = torch.sigmoid(a)
        x = torch.mm(a, x)
        return x, a