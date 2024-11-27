import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import make_lr_scheduler


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
    


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out # t + Graph

        out = F.softplus(out)
        return out

class ConditionalLinearModel(nn.Module):
    def __init__(self, nsteps, nfeatures: int, nblocks: int = 2, nunits: int = 128):
        super(ConditionalLinearModel, self).__init__()
        self.nsteps = nsteps
        self.inblock = ConditionalLinear(nfeatures, nunits, nsteps)
        self.midblock = nn.ModuleList([ConditionalLinear(nunits, nunits, nsteps) for _ in range(nblocks)])
        self.outblock = nn.Linear(nunits, nfeatures)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        val = self.inblock(x, t)
        for midblock in self.midblock:
            val = midblock(val, t)
        val = self.outblock(val)
        return val
    



def make_diffusion_model_optimizer_and_lr_scheduler(config, device = "cpu"):

    nblocks = 4
    nunits = 128
    model = ConditionalLinearModel(nsteps=config.diffusion_steps, nfeatures=2, nblocks=nblocks, nunits=nunits)
    is_model_trained = False

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # Linear decay function: lr = lr_init * (1 - t / total_epochs)
    lr_scheduler = make_lr_scheduler(optimizer = optimizer, config = config)

    if hasattr(config, 'load_model_chkpt_path') and config.load_model_chkpt_path is not None:
        try:
            chkpt = torch.load(config.load_model_chkpt_path)
            print('Loading diffusion-model checkpoint is successful.')

        except:
            print(f'Loading state-augmented training checkpoint from path {config.load_model_chkpt_path} failed.\nDefault weight initialization of the model, optimizer and lr scheduler is applied.')

        try:
            print('Model_state_dict: ', chkpt['model_state_dict'])
            model.load_state_dict(chkpt['model_state_dict'], strict = False)
            print('Loading pre-trained state-augmented model weights is successful.')
            is_model_trained = True
        except:
            print('Loading state dict of the hyperpolicy-diffusion-trainer model failed.\nDefault initialization of the model is applied.')

        try:
            optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            print('Loading hyperpolicy-diffusion-trainer optimizer state dict is successful.')
        except:
            print('Loading state dict of the hyperpolicy-diffusion-trainer optimizer failed.\nDefault initialization of the optimizer is applied.')

        try:
            lr_scheduler.load_state_dict(chkpt['scheduler_state_dict'])
            print('Loading hyperpolicy-diffusion-trainer learning scheduler state dict is successful.')
        except:
            print('Loading state dict of the diffusion hyperpolicy-diffusion-trainer lr scheduler failed.\nDefault initialization of the lr scheduler is applied.')


    # if config.model_load_path is not None:
    #     try:
    #         model.load_state_dict(torch.load(config.model_load_path))
    #         print(f'Loading pretrained model weights from {config.model_load_path} is successful.')
    #         is_model_trained = True

    #     except:
    #         print(f'Could not load model state dict from {config.model_load_path}! Training model from scratch.')

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, is_model_trained, optimizer, lr_scheduler