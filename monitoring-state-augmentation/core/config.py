from types import SimpleNamespace

config_dict = {"experiment_name": "dynamic-importance-sampling-tau=5",
               "num_states": 3,
               "num_episodes": 500,
               "num_timesteps": 100,
               "gamma": .99,
               "epsilon": 0.0,
               "exploration_temperature": 1.0,
               "batch_size_t": 5,
               "batch_size_lambdas": 50,
               "lambdas_max": 5.,
               "lr": 1e-4,
               "weight_decay": 1e-3,
               "tau": 10., # inverse parameter of kl-regularization strength (ideally infinite)
               "diffusion_steps": 200,
               "diffuse_n_samples": 40,
               "beta_schedule": "cosine",
               "model_load_path": None,
            #    "num_epochs": 20000,
               "chkpt_every_n_epochs": 100,
               "weight_decay": 0.0,
               "batch_size": 10,
               "verbose": 1 # 0 for no verbosity, 1 for limited verbosity, 2 for full verbosity
}

config = SimpleNamespace(**config_dict)