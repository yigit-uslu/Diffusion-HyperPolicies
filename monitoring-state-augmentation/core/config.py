from types import SimpleNamespace

num_episodes = num_epochs = 5000
num_iters_per_epoch = 10

batch_size_lambdas = 100 # 500

default_log_freq_rl = num_episodes // 100 # 20
default_log_freq_dm = num_episodes // 20

config_dict = {"experiment_name": "hyperpolicy-seq-importance-sampler-monday-evening-gamma=0.9-tau=50",
               "num_states": 3,
               "num_episodes": num_episodes,
               "num_timesteps": 100,
               "gamma": .9,
               "epsilon": 0.0,
               "exploration_temperature": 1.0,
               "batch_size_t": 20,
               "batch_size_lambdas": batch_size_lambdas,
               "lambdas_max": 5.0,
               "lambdas_sampler_alpha": 0.9,
               "dqn_num_features_list": [64, 64, 64],
               "lr": 1e-3,
               "weight_decay": 1e-3,
               "tau": 50., # inverse parameter of kl-regularization strength (ideally infinite)
               "tau_gamma": 1.05,
               "tau_step": 5,
               "diffusion_loss_thresh": 0.20,
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