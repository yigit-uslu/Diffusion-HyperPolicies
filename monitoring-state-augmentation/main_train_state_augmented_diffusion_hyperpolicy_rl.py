import torch
import tqdm
from core.Diffusion import DiffusionLearner
from core.env import BaseEnv, BaseEnv2
from core.policy import DQNAgent
from core.config import config
from utils.data_utils import LambdaNormalization, LambdaSampler, SequentialWeightedLambdaSampler, WeightedLambdaSampler, StateAugmentedLambdaSampler
from utils.logger_utils import LagrangiansImportanceSamplerLogger, make_dm_train_loggers
from utils.model_utils import make_diffusion_model_optimizer_and_lr_scheduler
from utils.utils import epsilon_decay_formula, temperature_decay_formula, make_kl_regularization_decay_scheduler


def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device; ', device)

    if config.env == 'v1':
        env = BaseEnv(device=device) # monitoring environment
    elif config.env == 'v2':
        env = BaseEnv2(device=device) # monitoring environment
    else:
        raise ValueError
    
    agent = DQNAgent(num_features_list=config.dqn_num_features_list, device=device) # agent


    hyperpolicy_learner = DiffusionLearner(config=config.diffusion_config, device=device)

    lambdas_norm = LambdaNormalization()
    
    dm_model, is_dm_model_trained, dm_optimizer, dm_lr_scheduler = \
        make_diffusion_model_optimizer_and_lr_scheduler(config=config.diffusion_config, device=device)
    
    tau_scheduler = make_kl_regularization_decay_scheduler(config=config, device=device)
    
    
    num_episodes = config.num_episodes # 1000
    num_timesteps = config.num_timesteps # 100
    n_lambdas = config.batch_size_lambdas # 5

    epsilon_init = agent.epsilon
    
    flip_symmetry = True
    uniform_lambdas_sampler = LambdaSampler(lambdas_max=config.lambdas_max, n_lambdas=len(env.states) - 1, lambdas_sparsity = config.lambdas_sparsity, device=device)
    uniform_lambdas_weights = torch.ones((n_lambdas,)).to(uniform_lambdas_sampler.device)

    lambdas = uniform_lambdas_sampler.sample(n_samples=n_lambdas, flip_symmetry=flip_symmetry) # reference Uniform distribution
    importance_weights = uniform_lambdas_weights.clone()

    lambdas_seq_importance_sampler = SequentialWeightedLambdaSampler(samples=lambdas, weights=importance_weights, alpha=config.lambdas_sampler_alpha, device=device)
    lagrangian_importance_sampler_logger = LagrangiansImportanceSamplerLogger(data = [], log_path=f"./logs/{config.experiment_name}/importance-sampler")
    
    sa_lambdas_sampler = StateAugmentedLambdaSampler(lambdas_max = config.lambdas_max, n_lambdas = len(env.states) - 1,
                                                     lr_lambdas=config.lr_lambdas,
                                                     device = device,
                                                     log_path = f"./logs/{config.experiment_name}/sa-rl-test"
                                                     ) # update


    for episode in tqdm.tqdm(range(num_episodes)):
        state, done = env.reset(n_states=n_lambdas)

        ### Sample state-augmenting dual multipliers ###
        # lambdas_importance_sampler = WeightedLambdaSampler(samples=lambdas, weights=torch.ones_like(importance_weights), device=device)
        # lambdas = lambdas_importance_sampler.sample(n_samples=n_lambdas, flip_symmetry=flip_symmetry)
        

        ############################ BEGIN: RL-Policy Update #################################
        total_reward = 0
        t = 0
        # done = False

        agent.policy_net.train()

        all_aug_rewards = []
        while not all(done) and t < num_timesteps:
            action_mask = env.get_invalid_action_mask(states=state)
            action, probs = agent.select_action((state, lambdas), action_mask)
            next_state, reward, done = env.step(action)
            agent.store_experience((state.clone(), lambdas.clone()), action, reward, next_state, done)
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
                                              decay_rate=config.epsilon_decay_rate, # 1.0
                                              epsilon_min=config.epsilon_min
                                              )
        
        agent.exploration_temperature = temperature_decay_formula(episode = episode,
                                                                  T_init = config.exploration_temperature,
                                                                  T_min = config.exploration_temperature_min, # .01,
                                                                  decay_rate=config.exploration_temperature_decay_rate # 0.99
                                                                  )
        
        agent.lr_scheduler.step()

        ############################ END: RL-Policy Update #################################


        if (episode + 1) % sa_lambdas_sampler.loggers[0].log_freq == 0:
            sa_lambdas = sa_lambdas_sampler.sample(episode=episode, n_samples=1, env=env, agent=agent)


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

        # lambdas_importance_sampler = WeightedLambdaSampler(samples=lambdas, weights=importance_weights, device=device)
        lambdas_seq_importance_sampler.update_samples(samples=lambdas, weights=importance_weights)
        

        dm_model, dm_optimizer, dm_lr_scheduler, dm_loss = hyperpolicy_learner.optimize(epoch=episode,
                                                                                        model=dm_model,
                                                                                        optimizer=dm_optimizer,
                                                                                        lr_scheduler=dm_lr_scheduler,
                                                                                        loss_fn=torch.nn.MSELoss(),
                                                                                        lambdas_sampler=lambdas_seq_importance_sampler,
                                                                                        lambdas_norm=lambdas_norm,
                                                                                        device=device
                                                                                        )
        
        dm_lr_scheduler.step()
        
        if dm_loss > config.diffusion_loss_thresh:
            print(f"Epoch = {episode}\tDm_loss = {dm_loss} > {config.diffusion_loss_thresh} = required threshold.")
            # lambdas = uniform_lambdas_sampler.sample(n_samples=n_lambdas, flip_symmetry=flip_symmetry) # reference Uniform distribution
            lambdas = lambdas_seq_importance_sampler.sample(n_samples=n_lambdas, flip_symmetry=flip_symmetry) # reference Uniform distribution

            ### Inspect diffusion model generations ###
            for logger in (logger for logger in hyperpolicy_learner.loggers if logger.log_metric == 'scatter-lambdas'):
                logger.update_data({"epoch": episode, "scatter-lambdas": (None, lambdas_seq_importance_sampler.sample(n_samples=lambdas.shape[0], flip_symmetry=False).detach().cpu().numpy())})
                logger()

            pass

        else:
            print(f"Epoch = {episode}\ttau = {tau}\tDm_loss = {dm_loss} < {config.diffusion_loss_thresh} = required threshold. Sampling dual multipliers from diffusion-hyperpolicy.")
            ### Generate lambdas to be sampled by the state-augmented rl-policy optimization in next epoch ###
            
            xgen, _ = hyperpolicy_learner.sample_ddpm(model=dm_model, nsamples=n_lambdas, nfeatures=lambdas.shape[-1], device=device)
            lambdas = lambdas_norm.reverse(xgen).to(device)

            ### Inspect diffusion model generations ###
            for logger in (logger for logger in hyperpolicy_learner.loggers if logger.log_metric == 'scatter-lambdas'):
                logger.update_data({"epoch": episode, "scatter-lambdas": (lambdas.detach().cpu().numpy(), lambdas_seq_importance_sampler.sample(n_samples=lambdas.shape[0], flip_symmetry=False).detach().cpu().numpy())})
                logger()

            lambdas = lambdas.data.clamp_(min = 0, max = config.lambdas_max)

        ############################ END: Diffusion Hyper-Policy Update #################################
            
        tau_scheduler.log_lr()

    print("Training finished.")
    



if __name__ == '__main__':
    main()