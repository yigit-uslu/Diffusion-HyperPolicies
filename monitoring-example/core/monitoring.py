from collections import defaultdict
import torch
import torch.nn as nn
import tqdm
from core.config import config

class MonitoringProblem(nn.Module):
    def __init__(self, env, agent, policy, device = 'cpu'):
        super().__init__()
        self.env = env
        self.agent = agent
        self.policy = policy
        self.device = device

    def eval_policy(self, n_episodes, n_timesteps, lambdas_list, lr_lambdas = 0.):
        all_variables_across_all_episodes = []
        for lambdas_0 in tqdm.tqdm(lambdas_list):
            for episode in range(n_episodes):
                state = self.env.reset()
                # lambdas_0 = lambdas_sampler.sample(n_samples = 1).squeeze(0)

                all_variables = defaultdict(list)
                for timestep in range(n_timesteps): # Limit timesteps as episode
                    # state = env.reset()
                    action = self.agent.select_action(augmented_state=(state, lambdas_0))
                    new_state, rewards, _ = self.env.step(action=action)

                    vars = {'timestep': timestep,
                            'old_state': (state, *lambdas_0.tolist()),
                            'new_state': (new_state, *lambdas_0.tolist()),
                            'action': action,
                            'rewards': rewards.tolist()
                            }
                    
                    for key in vars.keys():
                        all_variables[key].append(vars[key])

                    state = new_state
                    lambdas_0 -= lr_lambdas * (rewards[1:] - torch.tensor(self.policy.c).to(lambdas_0.device)[1:])
                    lambdas_0.data.clamp_(min = 0, max = config.lambdas_max)

                all_variables_across_all_episodes.append({'episode': episode, 'lambdas': lambdas_0, 'variables': all_variables})

        return all_variables_across_all_episodes
    

    def value_func(self, lambdas_list):

        if config.verbose >= 1:
            print(r'Computing $V(\pi^\star, \lambda)$ for a set of dual multipliers.')
        n_episodes = 10

        runs = self.eval_policy(n_episodes=n_episodes, n_timesteps=config.num_timesteps,
                                       lambdas_list=lambdas_list, lr_lambdas=0.
                                       )
        
        cum_rewards = torch.zeros(size = (len(lambdas_list), len(self.policy.c)), dtype = torch.float32, device = self.device)

        for i, lambdas in enumerate(lambdas_list):
            acc_reward = torch.stack([torch.tensor(run['variables']['rewards']) * (config.gamma)**torch.tensor(run['variables']['timestep']).unsqueeze(-1) for run in runs if torch.all(run['lambdas'] == lambdas)], dim = 0).mean(dim = (0, 1)).to(cum_rewards.device)
            cum_rewards[i] = acc_reward

        return cum_rewards
    

    def lagrangian(self, lambdas, values = None):
        if values is None:
            values = self.value_func([lam for lam in lambdas]).T.to(self.device)
        
        if config.verbose >= 1:
            print('values: ', values.shape)
            print('lambdas.shape')
        obj = values[0].to(self.device)
        constraint_slacks = (lambdas.T * (values[1:] - torch.tensor(self.policy.c, device = self.device)[1:].unsqueeze(-1))).sum(0)

        if config.verbose >= 1:
            print('constraint slacks: ', constraint_slacks.shape)
        lagrangian = obj + constraint_slacks

        return lagrangian
            
