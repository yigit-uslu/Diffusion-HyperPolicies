from types import SimpleNamespace

config_dict = {"num_states": 3,
               "num_episodes": 100,
               "num_timesteps": 50,
               "gamma": 1.,
               "lambdas_max": 5.,
               "tau": 10., # inverse parameter of kl-regularization strength (ideally infinite)
               "diffusion_steps": 200,
               "diffuse_n_samples": 40,
               "beta_schedule": "cosine",
               "model_load_path": None,
               "num_epochs": 20000,
               "chkpt_every_n_epochs": 100,
               "lr": 1e-4,
               "weight_decay": 0.0,
               "batch_size": 50,
               "verbose": 1 # 0 for no verbosity, 1 for limited verbosity, 2 for full verbosity
}

config = SimpleNamespace(**config_dict)