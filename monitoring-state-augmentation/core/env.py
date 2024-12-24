import numpy as np
import torch
import torch.nn as nn

from core.config import config
from utils.utils import has_len_attribute

class BaseEnv(nn.Module):
    def __init__(self, init_state = 0, device = 'cpu'):
        self.device = device
        self.states = torch.arange(0, config.num_states).to(device=device)
        self.rewards = torch.zeros_like(self.states)
        self.init_state = init_state
        self.state = torch.LongTensor([init_state]).to(device=self.device)
        self.done = False

    def reset(self, n_states = 1):
        self.state = torch.randint(low=self.states.min(), high=self.states.max() + 1, size = (n_states,)).to(device=self.device)
        self.done = [False for _ in range(n_states)]
        return self.state, self.done
    

    def step(self, actions):
        if not has_len_attribute(actions):
            actions = [actions]

        all_rewards = []
        for i, action in enumerate(actions):
            
            if self.done[i] and config.verbose >= 1:
                raise ValueError(f"Episode has ended for trajectory {i}.")

            # Define state-action-state transitions
            if self.state[i] == 0:
                if action == 0:
                    self.raise_invalid_action_error(action)
                else:
                    self.state[i] = action
                    rewards = torch.zeros_like(self.rewards)
                    rewards[self.state[i]] = 1
                    self.done[i] = False

            else:
                if action not in [0, self.state[i]]:
                    self.raise_invalid_action_error(action)

                else:
                    self.state[i] = action
                    rewards = torch.zeros_like(self.rewards)
                    rewards[self.state[i]] = 1
                    self.done[i] = False

            all_rewards.append(rewards)
        rewards = torch.stack(all_rewards, dim = 0).to(device=self.device)

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

        # print('all_mask[0].shape: ', all_mask[0].shape)
        return torch.stack(all_mask, dim = 0).to(device=self.device)
    



class BaseEnv2(nn.Module):
    "Self-loop at the objective is allowed."
    def __init__(self, init_state = 0, device = 'cpu'):
        self.device = device
        self.states = torch.arange(0, config.num_states).to(device=device)
        self.rewards = torch.zeros_like(self.states)
        self.init_state = init_state
        self.state = torch.LongTensor([init_state]).to(device=self.device)
        self.done = False

    def reset(self, n_states = 1):
        self.state = torch.randint(low=self.states.min(), high=self.states.max() + 1, size = (n_states,)).to(device=self.device)
        self.done = [False for _ in range(n_states)]
        return self.state, self.done
    

    def step(self, actions):
        if not has_len_attribute(actions):
            actions = [actions]

        all_rewards = []
        for i, action in enumerate(actions):
            
            if self.done[i] and config.verbose >= 1:
                raise ValueError(f"Episode has ended for trajectory {i}.")

            # Define state-action-state transitions
            if self.state[i] == 0:
                # if action == 0:
                #     self.raise_invalid_action_error(action)
                # else:
                self.state[i] = action
                rewards = torch.zeros_like(self.rewards)
                rewards[self.state[i]] = 1
                self.done[i] = False

            else:
                if action not in [0, self.state[i]]:
                    self.raise_invalid_action_error(action)

                else:
                    self.state[i] = action
                    rewards = torch.zeros_like(self.rewards)
                    rewards[self.state[i]] = 1
                    self.done[i] = False

            all_rewards.append(rewards)
        rewards = torch.stack(all_rewards, dim = 0).to(device=self.device)

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
                    # if action == 0:
                    #     mask = 1
                    # else:
                    mask = 0

                else:
                    if action not in [0, state]:
                        mask = 1
                    else:
                        mask = 0

                invalid_action_mask_arr.append(mask)

            all_mask.append(torch.LongTensor(invalid_action_mask_arr))

        # print('all_mask[0].shape: ', all_mask[0].shape)
        return torch.stack(all_mask, dim = 0).to(device=self.device)

        