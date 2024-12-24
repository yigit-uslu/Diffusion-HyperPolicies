import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from core.config import config, default_log_freq_rl
from utils.logger_utils import RLTestLambdaScatterLogger, RLTestActionsLogger, RLTestLambdasLogger, RLTestRewardsLogger, RLTestActionProbsLogger, RLTestStatesLogger


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
    def __init__(self, lambdas_max = config.lambdas_max, n_lambdas = config.num_states - 1, lambdas_sparsity = 0.0, device = 'cpu'):
        self.device = device
        self.lambdas_max = lambdas_max
        self.n_lambdas = n_lambdas
        self.lambdas_sparsity = lambdas_sparsity

    # def forward(self, n_samples):
    #     return self.sample(n_samples=n_samples)

    def sample(self, n_samples = 1, flip_symmetry = True):
        assert n_samples % 2 == 0 or n_samples == 1, "The number of samples should be an even number or 1."

        if flip_symmetry and not n_samples == 1:
            lambdas = self.lambdas_max * torch.rand(size = (n_samples // 2, self.n_lambdas), dtype = torch.float32, device=self.device)

            eps = 1. * (torch.rand_like(lambdas)[:, 0:1] < self.lambdas_sparsity)
            lambdas = (1 - eps) * lambdas + eps * torch.zeros_like(lambdas)

            perm = torch.randperm(lambdas.shape[0])
            lambdas_mirrored = torch.flip(lambdas, dims = (-1,))[perm].to(self.device)
            lambdas = torch.cat([lambdas, lambdas_mirrored], dim = 0)
        else:

            lambdas = self.lambdas_max * torch.rand(size = (n_samples, self.n_lambdas), dtype = torch.float32, device=self.device)
            
            eps = 1. * (torch.rand_like(lambdas)[:, 0:1] < self.lambdas_sparsity)
            lambdas = (1 - eps) * lambdas + eps * torch.zeros_like(lambdas)
            
        return lambdas
    


class WeightedLambdaSampler(LambdaSampler):
    def __init__(self, samples, weights, device = 'cpu'):
        super(WeightedLambdaSampler, self).__init__(device=device)
        # self.update_samples(samples=samples, weights=weights)
        self.unweighted_samples = samples 
        self.weights = weights

    def update_samples(self, samples, weights):
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
    


class SequentialWeightedLambdaSampler(WeightedLambdaSampler):
    def __init__(self, samples, weights, alpha = 0.1, max_stored_samples = config.batch_size_lambdas * 10, device='cpu'):
        super().__init__(samples, weights, device)
        self.alpha = alpha # new samples are equally as importnnt as the old ones when alpha = 0.5
        self.max_stored_samples = max_stored_samples

    def update_samples(self, samples, weights):
        self.unweighted_samples = torch.cat((self.unweighted_samples, samples), dim = 0)
        self.weights = torch.cat(((1 - self.alpha) * self.weights, self.alpha * weights), dim = 0)

        self.unweighted_samples = self.unweighted_samples[-self.max_stored_samples:] if len(self.unweighted_samples) > self.max_stored_samples else self.unweighted_samples
        self.weights = self.weights[-self.max_stored_samples:] if len(self.weights) > self.max_stored_samples else self.weights



class StateAugmentedLambdaSampler(LambdaSampler):
    def __init__(self, lambdas_max, n_lambdas, lr_lambdas = 0.5, device = 'cpu', log_path = None):
        super().__init__(lambdas_max=lambdas_max, n_lambdas=n_lambdas, device=device)
        self.log_path = log_path
        self.lr_lambdas = lr_lambdas

        self.loggers = [
                        RLTestLambdasLogger(data = [], log_path = self.log_path), 
                        RLTestRewardsLogger(data = [], log_path = self.log_path),
                        RLTestActionProbsLogger(data = [], log_path = self.log_path), 
                        RLTestStatesLogger(data = [], log_path = self.log_path),
                        RLTestActionsLogger(data = [], log_path = self.log_path),
                        RLTestLambdaScatterLogger(data = [], log_path = self.log_path)
                        # RLTestQNetworkLogger(data = [], log_path = self.log_path)
                        ]
        
        # for logger in self.loggers:
        #     logger.log_freq = default_log_freq_rl 


    def update_agent(self, agent):
        self.agent = agent


    def sample(self, env, agent, n_samples = 1, episode = None):
        # state-augmented evaluation
        agent.policy_net.eval()

        num_timesteps = 500

        n_lambdas = n_samples
        state, done = env.reset(n_states=n_lambdas)

        t = 0
        lambdas = torch.zeros((n_lambdas, self.n_lambdas)).to(self.device) # [self.n_lambdas = number of constraints]

        hist = []

        while not all(done) and t < num_timesteps:
            action_mask = env.get_invalid_action_mask(states=state)
            action, action_probs = agent.select_action((state, lambdas), action_mask)
            next_state, reward, done = env.step(action)
            hist.append(( (state.detach().clone(), lambdas.detach().clone()), action, action_probs, reward, next_state, done) )
            # ac_loss, aug_rewards = agent.optimize(epoch=episode)
            state = next_state
            # total_reward += reward
            t += 1

            reward_slacks = reward - agent.c.unsqueeze(0)
            # print('lambdas.shape: ', lambdas.shape)
            # print("rewards.shape: ", reward.shape)
            # print("reward_slacks.shape: ", reward_slacks.shape)
            lambdas -= self.lr_lambdas * reward_slacks[..., 1:]
            lambdas.data.clamp_(min = 0.0, max = self.lambdas_max)


        states, actions, action_probs, rewards, next_states, dones = zip(*hist)

        if isinstance(states, tuple): # state augmentation
            states, lambdas = zip(*states)
            lambdas = torch.stack(lambdas, dim = 0).to(dtype=torch.float32, device=self.device) # [T, n_lambdas, n_constraints]

        else:
            lambdas = None

        states = torch.stack(states, dim=0).to(dtype=torch.long, device=self.device) # [T, n_lambdas]
        actions = torch.stack(actions, dim=0).to(dtype=torch.long, device=self.device) # .view(-1, 1) # [T, n_lambdas]
        action_probs = torch.stack(action_probs, dim=0).to(dtype=torch.float32, device=self.device) # [T, n_lambdas, n_states = n_constraints + 1]
        rewards = torch.stack(rewards, dim=0).to(dtype=torch.float32, device=self.device) # [T, n_lambdas, n_states = n_constraints + 1]
        next_states = torch.stack(next_states, dim=0).to(dtype=torch.long, device=self.device)
        dones = torch.BoolTensor(dones).to(device=self.device) # [T, n_lambdas]

        variables = {'epoch': episode,
                     'lambdas': [lambdas[:, i].detach().cpu().numpy() for i in range(lambdas.shape[1])],
                     'rewards': [rewards[:, i].detach().cpu().numpy() for i in range(rewards.shape[1])],
                     'actions': [actions[:, i].detach().cpu().numpy() for i in range(actions.shape[1])],
                     'states': [states[:, i].detach().cpu().numpy() for i in range(states.shape[1])],
                     'action-probs': [action_probs[:, i].detach().cpu().numpy() for i in range(action_probs.shape[1])]
                     }

        for logger in self.loggers:

            if logger.log_metric in ['scatter-lambdas']:
                logger.update_data({"epoch": episode, "scatter-lambdas": (np.concatenate(variables['lambdas'], axis = 0), None)})
                logger(lambdas_label = ['SA-RL-sampled', None])
            else:
                logger.update_data(variables)
                logger()

        

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