import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from core.config import config


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        # Return the length of the shorter dataset
        return min([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        items = [dataset[idx] for dataset in self.datasets]
        return (*items,)


def importance_sampler(X_0, Y_0 = None, weights = None, batch_size = 1000, replacement = True):
    X_dataset = TensorDataset(X_0)
    if Y_0 is not None:
        Y_dataset = TensorDataset(Y_0)
        dataset = ZippedDataset([X_dataset, Y_dataset])
    else:
        dataset = X_dataset

    weights = torch.ones_like(len(X_dataset)) if weights is None else weights
    sampler = WeightedRandomSampler(weights, len(weights), replacement = replacement)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             sampler=sampler,
                                             batch_size = batch_size,
                                             drop_last = False
                                             )
    return dataloader



class LambdaNormalization(nn.Module):
    def __init__(self, x_min = 0, x_max = config.lambdas_max, x_target_min = -1/2, x_target_max = 1/2):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.x_target_min = x_target_min
        self.x_target_max = x_target_max

        "Linear transformation with slope m and offset b"
        self.m = (self.x_target_max - self.x_target_min) / (self.x_max - self.x_min)
        self.b = self.x_target_max - self.m * self.x_max
        assert np.abs(self.b - (self.x_target_min - self.m * self.x_min)) < 1e-6, 'Slope calculation error!'

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.m * x + self.b
    
    def reverse(self, x):
        "Invert the linear transformation"
        return LambdaNormalization(x_min=self.x_target_min,
                                   x_max=self.x_target_max,
                                   x_target_min=self.x_min,
                                   x_target_max=self.x_max
                                   )(x)
    


class LambdaSampler(nn.Module):
    def __init__(self, lambdas_max = config.lambdas_max, n_lambdas = config.num_states - 1, device = 'cpu'):
        self.device = device
        self.lambdas_max = lambdas_max
        self.n_lambdas = n_lambdas

    # def forward(self, n_samples):
    #     return self.sample(n_samples=n_samples)

    def sample(self, n_samples = 1, flip_symmetry = True):
        assert n_samples % 2 == 0 or n_samples == 1, "The number of samples should be an even number or 1."

        if flip_symmetry and not n_samples == 1:
            lambdas = self.lambdas_max * torch.rand(size = (n_samples // 2, self.n_lambdas), dtype = torch.float32, device=self.device)
            perm = torch.randperm(lambdas.shape[0])
            lambdas_mirrored = torch.flip(lambdas, dims = (-1,))[perm].to(self.device)
            lambdas = torch.cat([lambdas, lambdas_mirrored], dim = 0)
        else:
            lambdas = self.lambdas_max * torch.rand(size = (n_samples, self.n_lambdas), dtype = torch.float32, device=self.device)
            
        return lambdas
    


class WeightedLambdaSampler(LambdaSampler):
    def __init__(self, samples, weights, device = 'cpu'):
        super(WeightedLambdaSampler, self).__init__(device=device)
        self.unweighted_samples = samples 
        self.weights = weights

    def importance_sample(self, n_samples, replacement = True):

        dataloader = importance_sampler(X_0=self.unweighted_samples, # lambdas_orig
                                        Y_0=None, # no zipped dataset
                                        weights=self.weights,
                                        batch_size=n_samples,
                                        replacement=replacement
                                        )
        
        lambdas = next(iter(dataloader))[0]
        return lambdas
          

    def sample(self, n_samples = 1, flip_symmetry = True):

        assert n_samples % 2 == 0 or n_samples == 1, "The number of samples should be an even number or 1."

        if flip_symmetry and not n_samples == 1:
            lambdas = self.importance_sample(n_samples = n_samples // 2,
                                             replacement=True
                                             )
            
            perm = torch.randperm(lambdas.shape[0])
            lambdas_mirrored = torch.flip(lambdas, dims = (-1,))[perm].to(self.device)
            lambdas = torch.cat([lambdas, lambdas_mirrored], dim = 0)
        
        else:
            lambdas = self.importance_sample(n_samples = n_samples,
                                             replacement=True
                                             )
            
        return lambdas




# class WeightedLambdaSampler(LambdaSampler):
#     def __init__(self, weight_func, device = 'cpu'):
#         super(WeightedLambdaSampler, self).__init__(device=device)
#         self.weight_func = weight_func

#     def sample(self, dataloader = None, n_samples = 1, flip_symmetry = True, batch_size = 100, sample_single_batch = False):
#         if dataloader is None:
#             unweighted_samples = super().sample(n_samples=n_samples, flip_symmetry=flip_symmetry)
#             weights = self.weight_func(unweighted_samples)

#             dataloader = importance_sampler(X_0=unweighted_samples,
#                                             weights=weights,
#                                             batch_size=batch_size,
#                                             replacement=True
#                                             )
#         data = []
#         for i, temp in enumerate(dataloader):
#             data.append(temp[0])
#             if sample_single_batch:
#                 break
#         data = torch.cat(data, dim = 0)

#         return data, dataloader
    

# def make_weight_func(tau = 1.):
#     def weight_func(lambdas_all):
#         lagrangians_all = problem.lagrangian(lambdas=lambdas_all)
#         weights = (-tau * lagrangians_all).exp() / (-tau * lagrangians_all).exp().mean()

#         return weights
#     return weight_func