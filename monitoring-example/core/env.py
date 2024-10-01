import numpy as np
import torch
import torch.nn as nn

from core.config import NUM_STATES


class BaseEnv(nn.Module):
    def __init__(self, device = 'cpu'):
        self.device = device
        self.states = torch.arange(0, NUM_STATES).to(device=device)
        self.rewards = torch.zeros_like(self.states)
        self.state = 0
        self.done = False

    def reset(self):
        self.state = 0
        self.done = False
        return self.state
    

    def step(self, action):
        if self.done:
            raise ValueError("Episode has ended.")
        
        # Define state-action-state transitions
        if self.state == 0:
            if action == 0:
                self.raise_invalid_action_error(action)
            else:
                self.state = action
                rewards = torch.zeros_like(self.rewards)
                rewards[self.state] = 1
                self.done = False

        else:
            if action not in [0, self.state]:
                self.raise_invalid_action_error(action)

            else:
                self.state = action
                rewards = torch.zeros_like(self.rewards)
                rewards[self.state] = 1
                self.done = False

        return self.state, rewards, self.done



    def raise_invalid_action_error(self, action):
        raise ValueError(f"Action {action} is not allowed at state {self.state}.")

        