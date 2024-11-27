import torch
import tqdm
from core.Diffusion import DiffusionLearner
from core.env import BaseEnv
from core.policy import DQNAgent
from core.config import config
from utils.data_utils import LambdaNormalization, LambdaSampler, WeightedLambdaSampler
from utils.logger_utils import LagrangiansImportanceSamplerLogger, make_dm_loggers
from utils.model_utils import make_diffusion_model_optimizer_and_lr_scheduler
from utils.utils import epsilon_decay_formula, temperature_decay_formula, make_kl_regularization_decay_scheduler

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device; ', device)

    env = BaseEnv(device=device) # monitoring environment
    agent = DQNAgent(device=device) # agent


    hyperpolicy_learner = DiffusionLearner(config=config.diffusion_config, device=device)
    
    dm_model, is_dm_model_trained, dm_optimizer, dm_lr_scheduler = \
        make_diffusion_model_optimizer_and_lr_scheduler(config=config, device=device)
    
    tau_scheduler = make_kl_regularization_decay_scheduler(config=config, device=device)
    
    
    num_episodes = config.num_episodes # 1000
    num_timesteps = config.num_timesteps # 100
    n_lambdas = config.batch_size_lambdas # 5

    epsilon_init = agent.epsilon
    
    flip_symmetry = True
    uniform_lambdas_sampler = LambdaSampler(lambdas_max=config.lambdas_max, n_lambdas=len(env.states) - 1, device=device)
    uniform_lambdas_weights = torch.ones((n_lambdas,)).to(uniform_lambdas_sampler.device)

    lambdas = uniform_lambdas_sampler.sample(n_samples=n_lambdas, flip_symmetry=flip_symmetry) # reference Uniform distribution
    importance_weights = uniform_lambdas_weights.clone()

    lagrangian_importance_sampler_logger = LagrangiansImportanceSamplerLogger(data = [], log_path=f"./logs/{config.experiment_name}")

    for episode in tqdm.tqdm(range(num_episodes)):
        state, done = env.reset(n_states=n_lambdas)

        ### Sample state-augmenting dual multipliers ###
        lambdas_importance_sampler = WeightedLambdaSampler(samples=lambdas, weights=torch.ones_like(importance_weights), device=device)
        lambdas = lambdas_importance_sampler.sample(n_samples=n_lambdas, flip_symmetry=flip_symmetry)
        

        ############################ BEGIN: RL-Policy Update #################################
        total_reward = 0
        t = 0
        # done = False

        all_aug_rewards = []
        while not all(done) and t < num_timesteps:
            action_mask = env.get_invalid_action_mask(states=state)
            action = agent.select_action((state, lambdas), action_mask)
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

        ### Exploration and learning rate adjustments for Policy Optimizer ###
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
        # tau = episode * (10. / num_episodes)
        # lagrangians = torch.stack(all_aug_rewards, dim = 0).mean(0).view(n_lambdas, -1).mean(-1) # expected lagrangian
        # lambdas_weights = (-tau * lagrangians).exp() / (-tau * lagrangians).exp().mean()

        # weighted_avg_lagrangian = (lagrangians * lambdas_weights).mean()


        
        tau = tau_scheduler.get_lr(epoch=episode)  # KL-regularization parameter
        lagrangians = torch.stack(all_aug_rewards, dim = 0).mean(0).view(n_lambdas, -1).mean(-1) # expected lagrangian
        importance_weights = (-tau * lagrangians).exp() / (-tau * lagrangians).exp().mean()
        weighted_avg_lagrangian = (lagrangians * importance_weights).mean()

        lagrangian_importance_sampler_logger.update_data({'epoch': episode, 'importance-sampled-lagrangian': weighted_avg_lagrangian})
        lagrangian_importance_sampler_logger()

        lambdas_importance_sampler = WeightedLambdaSampler(samples=lambdas, weights=importance_weights, device=device)
        lambdas_norm = LambdaNormalization()
        # X = lambdas_norm(weighted_lambdas_all).to(device) # standardized dataset
        # X_0 = lambdas_norm.reverse(X).to(device) # normalized dataset

        dm_model, dm_optimizer, dm_lr_scheduler = hyperpolicy_learner.optimize(epoch=episode,
                                                                               model=dm_model,
                                                                               optimizer=dm_optimizer,
                                                                               lr_scheduler=dm_lr_scheduler,
                                                                               loss_fn=torch.nn.MSELoss(),
                                                                               sampler=lambdas_importance_sampler,
                                                                               lambdas_norm=lambdas_norm,
                                                                               device=device,
                                                                               loggers=None
                                                                               )
        
        xgen, _ = hyperpolicy_learner.sample_ddpm(model=dm_model, nsamples=n_lambdas, nfeatures=lambdas.shape[-1], device=device)
        lambdas = lambdas_norm.reverse(xgen).to(device)
        lambdas = lambdas.data.clamp_(min = 0)

        ############################ END: Diffusion Hyper-Policy Update #################################

    print("Training finished.")
    



if __name__ == '__main__':
    main()