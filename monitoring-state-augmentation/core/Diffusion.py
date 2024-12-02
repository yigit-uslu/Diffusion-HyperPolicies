import torch
import torch.nn as nn
import numpy as np
import abc
from argparse import Namespace
from utils.logger_utils import make_dm_train_loggers

from core.config import config

EXPERIMENT_NAME = config.experiment_name

class BaseDiffusionLearner(abc.ABC):
    def __init__(self, device, config):
        self.device = device
        self.beta_schedule = config.beta_schedule if hasattr(config, "beta_schedule") and config.beta_schedule is not None else "cosine"
        self.diffusion_steps = config.diffusion_steps if hasattr(config, "diffusion_steps") and config.diffusion_steps is not None else 500
        self.beta_min = config.beta_min if hasattr(config, "beta_min") and config.beta_min is not None else 0.1
        self.beta_max = config.beta_max if hasattr(config, "beta_max") and config.beta_max is not None else 20
        self.cosine_s = config.cosine_s if hasattr(config, "cosine_s") and config.cosine_s is not None else 0.008
        self.prior = torch.randn

        if self.beta_schedule == 'linear':
            self.betas = torch.linspace(self.beta_min / self.diffusion_steps, self.beta_max / self.diffusion_steps, self.diffusion_steps)
            self.alphas = 1 - self.betas
            self.baralphas = torch.cumprod(self.alphas, dim = 0)
            self.sde = None

        elif self.beta_schedule == 'cosine':
            # Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
            s = self.cosine_s
            timesteps = torch.tensor(range(0, self.diffusion_steps), dtype=torch.float32).to(self.device)
            schedule = torch.cos((timesteps / self.diffusion_steps + s) / (1 + s) * torch.pi / 2)**2

            self.baralphas = schedule / schedule[0]
            # betas = 1 - baralphas / torch.concatenate([baralphas[0:1], baralphas[0:-1]])
            self.betas = 1 - self.baralphas / torch.cat([self.baralphas[0:1], self.baralphas[0:-1]])
            self.alphas = 1 - self.betas
            self.sde = None

        else:
            raise NotImplementedError
        

    def sample_prior(self, sample_size):
        return self.prior(size=sample_size) # e.g., N(0, I)

        
    def noise(self, Xbatch, t):
        # eps = torch.randn(size=Xbatch.shape).to(Xbatch.device)
        device = Xbatch.device
        t = t.to(device)
        eps = self.sample_prior(sample_size=Xbatch.shape).to(device)
        noised = (self.baralphas.to(device)[t] ** 0.5).repeat(1, Xbatch.shape[1]) * Xbatch + ((1 - self.baralphas.to(device)[t]) ** 0.5).repeat(1, Xbatch.shape[1]) * eps
        return noised, eps
    

    def score_fnc_to_noise_pred(self, timestep, score, noise_pred = None):
        '''
        Convert score function to noise pred function and vice versa.
        '''
        sqrt_1m_alphas_cumprod = (1 - self.baralphas[timestep]).pow(0.5)
        if noise_pred is None:
            noise_pred = -sqrt_1m_alphas_cumprod * score
        else:
            score = -1 / sqrt_1m_alphas_cumprod  * noise_pred
        return score, noise_pred
    

    def sample_ddpm(self, model, nsamples, nfeatures, device = "cpu"):
        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
        with torch.no_grad():
            x = torch.randn(size=(nsamples, nfeatures)).to(device)
            # xt = [x]
            xt = [x.cpu()]
            scoret = []

            for t in range(self.diffusion_steps-1, 0, -1):
                predicted_noise = model(x, torch.full([nsamples, 1], t).to(device))
                # See DDPM paper between equations 11 and 12
                x = 1 / (self.alphas[t] ** 0.5) * (x - (1 - self.alphas[t]) / ((1-self.baralphas[t]) ** 0.5) * predicted_noise)
                if t > 1:
                    # See DDPM paper section 3.2.
                    # Choosing the variance through beta_t is optimal for x_0 a normal distribution
                    variance = self.betas[t]
                    std = variance ** (0.5)
                    x += std * torch.randn(size=(nsamples, nfeatures)).to(device)
                # xt += [x]
                xt += [x.cpu()]

                score, _ = self.score_fnc_to_noise_pred(timestep=t, score = None, noise_pred=predicted_noise)
                scoret += [score.cpu()]

            return x, {'x_t': xt, 'score_t': scoret}
            # return x, xt

    @abc.abstractmethod
    def optimize(self):
        pass


class DiffusionLearner(BaseDiffusionLearner):
    def __init__(self, config, device = 'cpu'):
        self.config = config
        self.device = device
        super(DiffusionLearner, self).__init__(device=device, config = config)

        self.loggers = make_dm_train_loggers(log_path=f"./logs/{EXPERIMENT_NAME}/dm-train")


    # Implement a training step.
    def optimize(self, epoch, model, optimizer, lr_scheduler, loss_fn, lambdas_sampler, lambdas_norm, device = 'cpu', **kwargs):
        
        epoch_loss = epoch_pgrad_norm = steps = 0
        for _ in range(self.config.num_iters_per_epoch):

            lambdas = lambdas_sampler.sample(n_samples = self.config.batch_size, flip_symmetry = False)
            Xbatch = lambdas_norm(lambdas)

            timesteps = torch.randint(0, self.diffusion_steps, size=[len(Xbatch), 1])
            noised, eps = self.noise(Xbatch, timesteps)
            predicted_noise = model(noised.to(device), timesteps.to(device))
            loss = loss_fn(predicted_noise, eps.to(device))

            optimizer.zero_grad()
            loss.backward()

            params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            pgrad_norm = np.sqrt(np.sum([p.grad.norm().item()**2 for p in params]))

            optimizer.step()

            epoch_loss += loss
            epoch_pgrad_norm += pgrad_norm
            steps += 1

        loss = epoch_loss / steps
        pgrad_norm = epoch_pgrad_norm / steps

        self.log({"epoch": epoch, "loss": loss.mean(), 'pgrad_norm': pgrad_norm.mean()})

        return model, optimizer, lr_scheduler, loss.mean().item()
    

    def log(self, log_variables):
        for logger in self.loggers:
            logger.update_data(log_variables)
            logger()



# class DiffusionLearner(nn.Module):
#     def __init__(self) -> None:
#         pass

#     def train():
#         loss_fn = nn.MSELoss()

#         optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
#         scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=nepochs)

#         dataloader = weighted_dataloader

#         all_variables = defaultdict(list)
#         for epoch in tqdm.tqdm(range(nepochs)):
#             epoch_loss = steps = 0
#             i = 0 
#             for data in dataloader:
#             # for i in range(0, len(X), batch_size):
#                 # Xbatch = X[i:i+batch_size]
#                 # Xbatch = Xbatch[0]
#                 data = data[0].to(device)
#                 Xbatch = lambdas_norm(data)

#                 perturb_sigma = 0.05
#                 Xbatch += perturb_sigma * torch.randn_like(Xbatch)
#                 timesteps = torch.randint(0, diffusion_steps, size=[len(Xbatch), 1])
#                 noised, eps = noise(Xbatch, timesteps)
#                 predicted_noise = model(noised.to(device), timesteps.to(device))
#                 loss = loss_fn(predicted_noise, eps.to(device))
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss
#                 steps += 1
#                 i += 1
            
#             epoch_loss = epoch_loss / steps
#             all_variables['loss'] = epoch_loss.item()