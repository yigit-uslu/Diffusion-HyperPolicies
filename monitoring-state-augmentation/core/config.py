from types import SimpleNamespace

num_episodes = num_epochs = 5000 # 5000
num_iters_per_epoch = 10

batch_size_lambdas = 50 # 500

default_log_freq_rl = num_episodes // 100 # 20
default_log_freq_dm = num_episodes // 20

config_dict = {"experiment_name": "hyperpolicy-seq-importance-sampler-dec-18-sa-rl-test-no-dm-lambdas-env1-always-explore-lambdas-sparsity=0.8",
               "env": "v1",
               "num_states": 3,
               "num_episodes": num_episodes,
               "num_timesteps": 100,
               "gamma": .99,
               "epsilon": 0.0,
               "epsilon_decay_rate": 1.0,
               "epsilon_min": 0.001,
               "exploration_temperature": 1.0,
               "exploration_temperature_decay_rate": 1.0,
               "exploration_temperature_min": 0.1,
               "batch_size_t": 20,
               "batch_size_lambdas": batch_size_lambdas,
               "lambdas_max": 5.0,
               "lambdas_sampler_alpha": 0.9,
               "lambdas_sparsity": 0.8,
               "dqn_num_features_list": [64, 64, 64, 64],
               "lr": 1e-4,
               "lr_lambdas": 0.1,
               "weight_decay": 1e-3,
               "tau": 10., # inverse parameter of kl-regularization strength (ideally infinite)
               "tau_gamma": 1.05,
               "tau_step": 5,
               "diffusion_loss_thresh": 0.02, # 0.2
               "diffusion_config":
               {
                 "diffusion_steps": 200,
                 "batch_size": batch_size_lambdas,
                 "num_epochs": num_epochs,
                 "num_iters_per_epoch": num_iters_per_epoch,
                 "diffuse_n_lambdas": batch_size_lambdas,
                 "beta_schedule": "cosine",
                 "load_model_chkpt_path": None,
                 "lr": 5e-4,
                 "weight_decay": 0.0
               }
}

config = SimpleNamespace(**config_dict)
config.diffusion_config = SimpleNamespace(**config.diffusion_config)