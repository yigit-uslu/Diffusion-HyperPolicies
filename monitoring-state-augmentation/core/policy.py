import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch_geometric.utils.repeat import repeat
from torch_scatter import scatter
from utils.model_utils import SinusoidalTimeEmbedding, AugmentedBlock
from utils.logger_utils import RLTrainAugRewardsLogger, RLTrainLossLogger, RLTrainPGradNormLogger, RLQTableLogger, RLTrainRewardsLogger

from core.config import config

EXPERIMENT_NAME = config.experiment_name


class OraclePolicy(nn.Module):
    def __init__(self, c = 1 / config.num_states,  device = 'cpu'):
        super(OraclePolicy, self).__init__()
        self.c = repeat(c, config.num_states)
        self.device = device

    def weighted_reward(self, x, lambdas):
        c_padded = torch.cat((0, self.c))
        lambdas_padded = torch.cat((1, lambdas))        

    def forward(self, x, lambdas):

        assert len(lambdas) == 2

        if torch.all(lambdas < 1):
            if config.verbose >= 2:
                print('Optimal deterministic policy maximizes time spent at state 0.')
            if x == 0:
                action = np.random.choice([1, 2]).item() # wlog choose between a nonzero action.
            else:
                action = 0


        elif torch.all(lambdas == 1):
            if config.verbose >= 2:
                print('All deterministic policies are optimal.')
            if x == 0:
                action = 1
            elif x == 1:
                action = 0
            elif x == 2:
                action = 2


        elif lambdas[0] > lambdas[1] and lambdas[0] > 1:
            if config.verbose >= 2:
                print('Optimal deterministic policy maximizes time spent at state 1.')
            if x in [0, 1]:
                action = 1
            elif x == 2:
                action = 0


        elif lambdas[1] > lambdas[0] and lambdas[1] > 1:
            if config.verbose >= 2:
                print('Optimal deterministic policy maximizes time spent at state 2.')
            if x in [0, 2]:
                action = 2
            elif x == 1:
                action = 0
            
        probs = torch.zeros(size=(len(self.c),), device=self.device, dtype = torch.float32)
        probs[action] = 1
        return probs


    

class DQNPolicy(nn.Module):
    def __init__(self, n_observations = config.num_states, n_actions = config.num_states, num_features_list = [64, 64], device = 'cpu'):
        super().__init__()
        self.device = device
        self.num_features_list =  num_features_list

        self.act = nn.LeakyReLU()
        self.state_embedding = nn.Sequential(nn.Embedding(n_observations, num_features_list[0], device=device), nn.Flatten(start_dim=1))
        self.lambdas_embedding = nn.Sequential(SinusoidalTimeEmbedding(n_channels=num_features_list[0]), nn.Linear(num_features_list[0], num_features_list[0], bias=False, device=device))

        self.aug_layers = nn.ModuleList()
        for i in range(len(num_features_list) - 1):
            self.aug_layers.append(AugmentedBlock(in_channels=num_features_list[i],
                                                  out_channels=num_features_list[i+1],
                                                  act = self.act)
                                                  )

        self.out = nn.Linear(self.aug_layers[-1].out_channels, n_actions)
        # self.out_mu = nn.Linear(self.fc_layers[-1].out_features, n_actions)
        # self.out_std = nn.Linear(self.fc_layers[-1].out_features, config.num_states)


    def forward(self, x, lambdas):

        # print(f"x.shape: {x.shape}\tlambdas.shape: {lambdas.shape}")

        x_embed = self.state_embedding(x) # [n_lambdas, 64]
        lambdas_embed = self.lambdas_embedding(lambdas) # [n_lambdas, 64]

        # print(f"x.shape: {x.shape} \t x_embed.shape: {x_embed.shape}")
        # print(f"lambdas.shape: {lambdas.shape} \t lambdas_embed.shape: {lambdas_embed.shape}")

        for i, layer in enumerate(self.aug_layers):
            x_embed = layer(x_embed, lambdas_embed)
        
        return self.out(x_embed)





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

    

class DQNAgent(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.device = device
        self.policy_net = DQNPolicy(device=device).to(device=self.device)
        self.target_net = DQNPolicy(device=device).to(device=self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.c = torch.FloatTensor([0, 1/3, 1/3]).to(device=device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)

        # Linear decay function: lr = lr_init * (1 - t / total_epochs)
        lr_lambda = lambda epoch: 1 - epoch / config.num_episodes

        self.lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lr_lambda)

        
        self.memory = []  # A simple list to store experience tuples
        self.gamma = config.gamma # 0.99  # Discount factor for future rewards
        self.epsilon = config.epsilon # 0.1  # Exploration probability
        self.exploration_temperature = config.exploration_temperature # 1.0  # exploration probability
        self.batch_size = config.batch_size_t # 20 # Batch size

    # def select_action(self, state, lambdas):
    #     if random.random() < self.epsilon:
    #         return random.randint(0, config.num_states-1)  # Random action (exploration)
    #     else:
    #         # state_tensor = torch.FloatTensor([state])
    #         # lambdas_tensor = torch.FloatTensor(lambdas)
    #         q_values = self.policy_net(state.to(self.device), lambdas.to(self.device))
    #         return torch.argmax(q_values).item()  # Best action (exploitation)
        
        self.loggers = []
        logger = RLTrainLossLogger(data = [], log_path=f"./logs/{EXPERIMENT_NAME}")
        self.loggers.append(logger)

        logger = RLTrainAugRewardsLogger(data = [], log_path = f"./logs/{EXPERIMENT_NAME}")
        self.loggers.append(logger)

        logger = RLTrainRewardsLogger(data = [], log_path = f"./logs/{EXPERIMENT_NAME}")
        self.loggers.append(logger)

        logger = RLTrainPGradNormLogger(data = [], log_path = f"./logs/{EXPERIMENT_NAME}")
        self.loggers.append(logger)

        logger = RLQTableLogger(data = [], log_path = f"./logs/{EXPERIMENT_NAME}")
        self.loggers.append(logger)
        

    def select_action(self, augmented_state, action_mask = None):

        state, lambdas = augmented_state
        if isinstance(state, torch.Tensor):
            state_tensor = state.clone().to(self.device)
        else:
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
        Q_values = self.policy_net(x = state_tensor, lambdas = lambdas) # [n_lambdas, n_actions=n_states]

        action = self.softmax_selection(Q_values=Q_values, mask=action_mask)

        eps = 1. * (torch.rand(size=action.size(), device = action.device) < self.epsilon) # boolean flag for random actions
        # random_action = torch.randint_like(input=eps, low=0, high=config.num_states)
        random_action = self.softmax_selection(Q_values=torch.ones_like(Q_values), mask=action_mask)

        action = (1. - eps) * action + eps * random_action

        return action
    
    
    def softmax_selection(self, Q_values, mask = None):


        temperature = self.exploration_temperature

        # print('Q_values.shape: ', Q_values.shape, Q_values.device)
        # print('mask.shape: ', mask.shape, mask.device) if mask is not None else None

        if mask is not None:
            # Apply mask: set invalid actions to a very low value
            Q_values_masked = Q_values + (mask.to(device=Q_values.device) * -1e10)
        else:
            Q_values_masked = Q_values

        Q_values_masked_max = Q_values_masked.max(dim = -1, keepdim = True)[0]
        Q_values_masked_stabilized = Q_values_masked - Q_values_masked_max
        
        # Softmax transformation (with temperature scaling)
        exp_Q = torch.exp(Q_values_masked_stabilized / temperature)
        # exp_Q = torch.exp(Q_values_masked / temperature)
        prob = exp_Q / torch.sum(exp_Q, dim = -1, keepdim=True)

        # print('prob.shape: ', prob.shape, prob.device)

        assert not torch.any(torch.isnan(prob)), print("Action probabilities have NaN values: ", prob)
        assert not torch.any(torch.isinf(prob)), print("Action probabilities have InF values: ", prob)

        # Select actions based on the softmax distribution, for each state in the batch
        actions = torch.multinomial(prob, 1)  # Sample actions for each row (state) in the batch
        
        # Select action based on probabilities
        return actions.squeeze(-1) # Remove the extra dimension, returning a 1D tensor of actions
    
        
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def sample_batch(self):
        return random.sample(self.memory, self.batch_size)
    

    def optimize(self, epoch = None):
        if len(self.memory) < self.batch_size:
            return None, None
        
        if not len(self.memory) % self.batch_size == 0:
            return None, None

        batch = self.sample_batch()
        states, actions, rewards, next_states, dones = zip(*batch)

        if isinstance(states, tuple): # state augmentation
            states, lambdas = zip(*states)
            lambdas = torch.stack(lambdas, dim = 0).to(dtype=torch.float32, device=self.device) # [T, n_lambdas, n_constraints]

        else:
            lambdas = None

        states = torch.stack(states, dim=0).to(dtype=torch.long, device=self.device) # [T, n_lambdas]
        actions = torch.stack(actions, dim=0).to(dtype=torch.long, device=self.device) # .view(-1, 1) # [T, n_lambdas]
        rewards = torch.stack(rewards, dim=0).to(dtype=torch.float32, device=self.device) # [T, n_lambdas, n_states = n_constraints + 1]
        next_states = torch.stack(next_states, dim=0).to(dtype=torch.long, device=self.device)
        dones = torch.BoolTensor(dones).to(device=self.device) # [T, n_lambdas]

        # Get Q values for current states and actions
        q_values_orig = self.policy_net(states.view(-1), lambdas.view(-1, lambdas.shape[-1]))
        # q_values = q_values.gather(1, actions.view(-1))
        q_values = q_values_orig[torch.arange(q_values_orig.shape[0]), actions.view(-1)]

        # Get Q values for next states from target network
        next_q_values = self.target_net(next_states.view(-1), lambdas.view(-1, lambdas.shape[-1]))
        next_q_values = next_q_values.max(dim = 1)[0].detach()

        # Compute target Q values
        aug_lambdas = torch.cat([torch.ones_like(lambdas[..., 0:1]), lambdas], dim = -1)
        aug_rewards = torch.sum(aug_lambdas * (rewards - self.c.unsqueeze(0).unsqueeze(0)), dim = -1).view(-1)
        target_q_values = aug_rewards + (self.gamma * next_q_values * ~dones.view(-1))

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values) # q_values.squeeze()

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()

        params = [p for p in self.policy_net.parameters() if p.grad is not None and p.requires_grad]
        pgrad_norm = np.sqrt(np.sum([p.grad.norm().item()**2 for p in params]))
        
        self.optimizer.step()

        # Periodically update the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())


        # Create a Q-table
        Q = np.zeros((config.num_states, config.num_states))
        for state in range(Q.shape[0]):
            for action in range(Q.shape[1]):
                state_action_index = 1. * (actions.view(-1) == action) * 1. * (states.view(-1) == state)
                try:
                    q = scatter(src=q_values_orig, index=state_action_index.to(torch.long), dim=0, reduce="mean")[1]
                    Q[state, action] = q[action].detach().cpu().numpy()
                except:
                    Q[state, action] = 0.

        

        # print('Q(s, a, \lambda = 0): ', Q)

        self.log({'epoch': epoch , 'Q_values': Q, 'loss': loss.mean(), 'pgrad_norm': pgrad_norm.mean(), 'aug_rewards': aug_rewards.mean(), 'rewards': rewards.mean(dim = (0, 1)).detach().cpu().numpy()})


        return loss, aug_rewards


    def log(self, log_variables):

        for logger in self.loggers:
            logger.update_data(log_variables)
            logger()




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
    
