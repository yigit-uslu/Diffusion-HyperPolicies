from types import SimpleNamespace

num_episodes = num_epochs = 1000
num_iters_per_epoch = 5

batch_size_lambdas = 100

config_dict = {"experiment_name": "hyperpolicy",
               "num_states": 3,
               "num_episodes": num_episodes,
               "num_timesteps": 100,
               "gamma": .99,
               "epsilon": 0.0,
               "exploration_temperature": 1.0,
               "batch_size_t": 20,
               "batch_size_lambdas": batch_size_lambdas,
               "lambdas_max": 5.,
               "lr": 5e-4,
               "weight_decay": 1e-3,
               "tau": 10., # inverse parameter of kl-regularization strength (ideally infinite)
               "diffusion_config":
               {
                 "diffusion_steps": 50,
                 "batch_size": batch_size_lambdas,
                 "num_epochs": num_epochs,
                 "num_iters_per_epoch": num_iters_per_epoch,
                 "diffuse_n_lambdas": batch_size_lambdas,
                 "beta_schedule": "cosine",
                 "load_model_chkpt_path": None,
                 "lr": 1e-4,
                 "weight_decay": 1e-3
               }
}

config = SimpleNamespace(**config_dict)