import torch
import tqdm
from core.env import BaseEnv
from core.policy import DQNAgent
from core.config import config
from utils.data_utils import LambdaSampler, importance_sampler
from utils.logger_utils import LagrangiansImportanceSamplerLogger
from utils.utils import epsilon_decay_formula, temperature_decay_formula

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device; ', device)

    env = BaseEnv(device=device)
    agent = DQNAgent(device=device)

    num_episodes = config.num_episodes # 1000
    num_timesteps = config.num_timesteps # 100
    n_lambdas = config.batch_size_lambdas # 5

    epsilon_init = agent.epsilon
    
    flip_symmetry = True

    # Training loop

    lambdas_sampler = LambdaSampler(lambdas_max=config.lambdas_max, n_lambdas=len(env.states) - 1, device=device)
    lambdas_weights = torch.ones((n_lambdas,)).to(lambdas_sampler.device)

    importance_sampler_logger = LagrangiansImportanceSamplerLogger(data = [], log_path=f"./logs/{config.experiment_name}")

    lambdas = lambdas_sampler.sample(n_samples=n_lambdas, flip_symmetry=flip_symmetry) # reference Unif distribution


    for episode in tqdm.tqdm(range(num_episodes)):
        state, done = env.reset(n_states=n_lambdas)
        
        # lambdas = torch.zeros((n_lambdas, 2)).to(device)
        # lambdas = torch.cat( [config.lambdas_max * torch.zeros((n_lambdas, 1)), torch.ones((n_lambdas, 1))], dim = -1).to(device)
        
        # lambdas = lambdas_sampler.sample(n_samples=n_lambdas, flip_symmetry=False) # reference Unif distribution
        
        lambdas_star_dataloader = importance_sampler(X_0=lambdas, # lambdas_orig
                                        #    Y_0=lagrangians_all,
                                               Y_0=None,
                                               weights=lambdas_weights, # unfiorm initially
                                               batch_size=n_lambdas // 2 if flip_symmetry else n_lambdas,
                                               replacement=True
                                               )
        # for data in lambdas_star_dataloader:
            # lambdas = data[0]
        lambdas = next(iter(lambdas_star_dataloader))[0]

        if flip_symmetry:
            perm = torch.randperm(lambdas.shape[0])
            lambdas_mirrored = torch.flip(lambdas, dims = (-1,))[perm].to(lambdas.device)
            lambdas = torch.cat([lambdas, lambdas_mirrored], dim = 0)

        # print("lambdas.shape: ", lambdas.shape)
        

        ############################ BEGIN: RL-Policy Update #################################
        total_reward = 0
        t = 0
        # done = False

        all_aug_rewards = []
        while not all(done) and t < num_timesteps:
            action_mask = env.get_invalid_action_mask(states=state)
            action, probs = agent.select_action((state, lambdas), action_mask)
            next_state, reward, done = env.step(action)
            agent.store_experience((state, lambdas), action, reward, next_state, done)
            ac_loss, aug_rewards = agent.optimize(epoch=episode)
            state = next_state
            total_reward += reward
            t += 1

            if aug_rewards is not None:
                all_aug_rewards.append(aug_rewards)


        # if episode % 100 == 0:
        #     print(f"Episode {episode}, Average total Reward: {(1. * total_reward).mean().item()}")

        agent.epsilon = epsilon_decay_formula(episode=episode,
                                              epsilon_init=epsilon_init,
                                              decay_rate=0.99
                                              )
        
        agent.exploration_temperature = temperature_decay_formula(episode = episode,
                                                                  T_init = 1.,
                                                                  T_min = .1,
                                                                  decay_rate=0.99
                                                                  )
        
        agent.lr_scheduler.step()

        ############################ END: RL-Policy Update #################################




        ############################ BEGIN: Diffusion Hyper-Policy Update #################################
        tau = episode * (10. / num_episodes)
        lagrangians = torch.stack(all_aug_rewards, dim = 0).mean(0).view(n_lambdas, -1).mean(-1) # expected lagrangian
        lambdas_weights = (-tau * lagrangians).exp() / (-tau * lagrangians).exp().mean()

        weighted_avg_lagrangian = (lagrangians * lambdas_weights).mean()

        importance_sampler_logger.update_data({'epoch': episode, 'importance-sampled-lagrangian': weighted_avg_lagrangian})
        importance_sampler_logger()

        # diffusion_loss, diffusion_weights = diffusion_learner.optimize()

        


        ############################ END: Diffusion Hyper-Policy Update #################################

    print("Training finished.")
    



if __name__ == '__main__':
    main()