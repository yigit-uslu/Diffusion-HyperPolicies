import numpy as np
import torch
import torch.nn as nn
from utils.utils import has_len_attribute

from core.config import config


class BaseEnv(nn.Module):
    def __init__(self, init_state = 0, device = 'cpu'):
        self.device = device
        self.states = torch.arange(0, config.num_states).to(device=device)
        self.rewards = torch.zeros_like(self.states)
        self.init_state = init_state
        self.state = init_state
        self.done = False

    def reset(self):
        self.state = torch.randint(low=self.states.min(), high=self.states.max() + 1, size = (1,))
        self.done = False
        return self.state
    

    def step(self, action):
        if self.done and config.verbose >= 1:
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
        if config.verbose >= 1:
            raise ValueError(f"Action {action} is not allowed at state {self.state}.")
        

    def get_invalid_action_mask(self, states):

        if not has_len_attribute(states):
            states = [states]

        all_mask = []
        for state in states:
            invalid_action_mask_arr = []
            for action in range(len(self.states)):
                if state == 0:
                    if action == 0:
                        mask = 1
                    else:
                        mask = 0

                else:
                    if action not in [0, state]:
                        mask = 1
                    else:
                        mask = 0

                invalid_action_mask_arr.append(mask)

            all_mask.append(torch.LongTensor(invalid_action_mask_arr))

        print('all_mask[0].shape: ', all_mask[0].shape)
        return torch.stack(all_mask, dim = 0).to(device=self.device)

            



        