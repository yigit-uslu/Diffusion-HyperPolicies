import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils.repeat import repeat

from core.config import NUM_STATES, VERBOSE, LAMBDAS_MAX


class OraclePolicy(nn.Module):
    def __init__(self, c = 1 / NUM_STATES,  device = 'cpu'):
        super(OraclePolicy, self).__init__()
        self.c = repeat(c, NUM_STATES)
        self.device = device

    def weighted_reward(self, x, lambdas):
        c_padded = torch.cat((0, self.c))
        lambdas_padded = torch.cat((1, lambdas))        

    def forward(self, x, lambdas):

        assert len(lambdas) == 2

        if torch.all(lambdas < 1):
            if VERBOSE >= 2:
                print('Optimal deterministic policy maximizes time spent at state 0.')
            if x == 0:
                action = np.random.choice([1, 2]).item() # wlog choose between a nonzero action.
            else:
                action = 0


        elif torch.all(lambdas == 1):
            if VERBOSE >= 2:
                print('All deterministic policies are optimal.')
            if x == 0:
                action = 1
            elif x == 1:
                action = 0
            elif x == 2:
                action = 2


        elif lambdas[0] > lambdas[1] and lambdas[0] > 1:
            if VERBOSE >= 2:
                print('Optimal deterministic policy maximizes time spent at state 1.')
            if x in [0, 1]:
                action = 1
            elif x == 2:
                action = 0


        elif lambdas[1] > lambdas[0] and lambdas[1] > 1:
            if VERBOSE >= 2:
                print('Optimal deterministic policy maximizes time spent at state 2.')
            if x in [0, 2]:
                action = 2
            elif x == 1:
                action = 0
            
        probs = torch.zeros(size=(len(self.c),), device=self.device, dtype = torch.float32)
        probs[action] = 1
        return probs



class Agent(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.device = device
        self.policy_net = OraclePolicy(device=device)
        

    def select_action(self, augmented_state):
        state, lambdas = augmented_state
        if isinstance(state, torch.Tensor):
            state_tensor = state.clone().to(self.device)
        else:
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(x = state_tensor, lambdas = lambdas)
        action = np.random.choice(len(probs.detach().cpu().numpy().flatten()), p = probs.detach().cpu().numpy().flatten())

        return action
    


class LambdaSampler(nn.Module):
    def __init__(self, lambdas_max = LAMBDAS_MAX, n_lambdas = NUM_STATES - 1, device = 'cpu'):
        self.device = device
        self.lambdas_max = lambdas_max
        self.n_lambdas = n_lambdas

    # def forward(self, n_samples):
    #     return self.sample(n_samples=n_samples)

    def sample(self, n_samples = 1):
        lambdas = self.lambdas_max * torch.rand(size = (n_samples, self.n_lambdas), dtype = torch.float32, device=self.device)
        return lambdas
