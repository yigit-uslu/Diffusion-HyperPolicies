import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SinusoidalTimeEmbedding(nn.Module):
    """
    https://nn.labml.ai/diffusion/ddpm/unet.html 
    """
    def __init__(self, n_channels: int, device = 'cpu'):
        super().__init__()
        self.n_channels = n_channels # scaled by 2 since there are 2 dual multipliers.
        self.act = Swish()
        self.device = device

        n_lambdas = 2

        self.lin_embed = nn.Sequential(nn.Flatten(start_dim=-2),
                                       nn.Linear(n_lambdas * self.n_channels // 4, self.n_channels, device=device),
                                       self.act,
                                       nn.Linear(self.n_channels, self.n_channels, device = device)
                                       )
        
    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = torch.log(torch.Tensor([10000.])) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = t.device) * -emb.to(t.device))
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim = -1)

        emb = self.lin_embed(emb) # [, n_lambdas, 16] -> [, 64]
        return emb
    


class AugmentedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        self.x_embed = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.aug_embed = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, aug = None):
        x = self.x_embed(x) 
        if aug is not None:
            x += self.aug_embed(aug)
        return self.act(x)