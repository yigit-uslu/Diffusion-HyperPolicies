import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils.repeat import repeat

from core.config import NUM_STATES


class OraclePolicy(nn.Module):
    def __init__(self, c = 1 / NUM_STATES,  device = 'cpu'):
        super(OraclePolicy, self).__init__()
        self.c = repeat(c, NUM_STATES)
        self.device = device

    def weighted_reward(self, x, lambdas):
        c_padded = torch.cat((0, self.c))
        lambdas_padded = torch.cat((1, lambdas))

        


    def forward(self, x, lambdas):
        return 0



class Agent(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.device = device
        self.policy_net = OraclePolicy(device=device)
        

    def select_action(self, state, lambdas):
        state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(x = state_tensor, lambdas = lambdas)

        action = np.random.choice(len(probs.flatten()), p = probs.detach().numpy().flatten())
        return action