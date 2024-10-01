from collections import defaultdict
import copy
import math
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
import os
from numpy import linalg as LA
from torch_geometric.data import Data, Dataset
import tqdm
from sklearn.datasets import make_swiss_roll



def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=0.5e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.data = X
        
    def __getitem__(self, index):
        # print(index)
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)

# print('len(X): ', len(X))


def make_logger(debug = False):
    if debug is True:
        def logger(x):
            print(x)
    else:
        def logger(x):
            pass

    return logger


    
def make_eval_obj_fnc(config):
    def eval_obj_fnc(avg_rates):
        return torch.sum(-avg_rates, dim = -1)

    def slack_to_metric(slack):
        return -slack / config.n
    slack_to_metric.metric_name = 'Mean rate (bps/Hz)'

    eval_obj_fnc.slack_to_metric = slack_to_metric

    return eval_obj_fnc


def make_eval_constraints_fnc(config):
    def eval_constraints_fnc(avg_rates):
        return config.r_min - avg_rates
    
    def slack_to_metric(slack):
        return -slack + config.r_min
    slack_to_metric.metric_name = 'rate (bps/Hz)'
    slack_to_metric.constraint_level = config.r_min
    
    eval_constraints_fnc.slack_to_metric = slack_to_metric
    
    return eval_constraints_fnc


def make_eval_resilient_gradient_fnc(config):
    # quadratic cost
    def eval_resilient_gradient_fnc(u):
        return config.alpha * u
    
    return eval_resilient_gradient_fnc


def load_RRM_dataset(dataset_path, save_path = None, plot_networks_list = [], logger = None, data_fabrication_factor = None, standardize = False):
    # fabrication_factor = args.RRM_data_fabrication_factor if hasattr(args, 'RRM_data_fabrication_factor') and args.RRM_data_fabrication_factor is not None else None
    fabrication_factor = data_fabrication_factor
    fabricate_data = (fabrication_factor is not None)
    
    dataset = torch.load(dataset_path)
    Ps = dataset['Ps']
    lambdas = dataset['lambdas']

    Ps = Ps / Ps.max().item()

    n_algos, n_networks, n_clients, n_trajectories, T = Ps.shape
    n_algos, n_networks, n_clients, n_trajectories, K = lambdas.shape

    logger(f'Ps: {Ps.shape}')

    Ps = Ps[-1].reshape(n_networks, n_clients, -1)
    lambdas = lambdas[-1].reshape(n_networks, n_clients, -1)

    if fabricate_data:
        if logger is not None:
            logger(f'Original dataset shape = {Ps.shape}')
        Ps_fabricated = fabricate_RRM_data(torch.from_numpy(Ps), nsamples=int((fabrication_factor-1) * n_trajectories * T), method='perturbation', perturbation_sigma=0.01, combine_orig_and_fabricated_data=False).detach().cpu().numpy()

        if logger is not None:
            logger(Ps_fabricated.shape)

        if save_path is not None and plot_networks_list is not None and len(plot_networks_list):
            for network_id in plot_networks_list:
                fig, ax = plt.subplots(1, 1, figsize = (6, 6))
                ax.scatter(Ps_fabricated[network_id, 0], Ps_fabricated[network_id, 1], alpha=0.2, label = 'Fabricated data')
                ax.scatter(Ps[network_id, 0], Ps[network_id, 1], marker="*", alpha=0.2, label = 'Original data')
                ax.set_title(f'Network #{network_id}')
                ax.legend(loc = 'best')

                plt.savefig(f'{save_path}/RRM_data_network_{network_id}.pdf', dpi = 300)
                plt.close(fig)

        Ps = np.concatenate((Ps, Ps_fabricated), axis = -1)

        if logger is not None:
            logger(f'Fabrication factor = {fabrication_factor}')
            logger(f'Post-fabrication dataset shape = {Ps.shape}')

    X = torch.from_numpy(Ps)
    # X = (X - X.mean()) / X.std() if standardize else X

    X_train = []
    for id in range(n_networks):
        x = torch.cat([X[id, client].reshape(-1, 1) for client in range(n_clients)], dim = -1)
        x = (x - x.mean()) / x.std() if standardize else x
        X_train.append(x)
    # X_train = [torch.cat((X[id, 0].reshape(-1, 1), X[id, 1].reshape(-1, 1),
    #                       X[id, 2].reshape(-1, 1), X[id, 3].reshape(-1, 1),
    #                       X[id, 4].reshape(-1, 1), X[id, 5].reshape(-1, 1),
    #                       X[id, 6].reshape(-1, 1), X[id, 7].reshape(-1, 1),
    #                       X[id, 8].reshape(-1, 1), X[id, 9].reshape(-1, 1),
    #                       X[id, 10].reshape(-1, 1), X[id, 11].reshape(-1, 1),), dim = -1) for id in range(n_networks)] # list of Tensor(-1, 1)

    return X_train, Ps


def load_swissroll_dataset(data_shape = None, save_path = None, plot_networks_list = [], logger = None, **kwargs):
    '''
    Make swissroll spiral for each two consecutive dimensions.
    '''
    noise = kwargs.get('noise', 0.5)
    active_network_idx = kwargs.get('active_network_idx', None)

    n_networks, n_clients, T = data_shape

    active_network_idx = range(n_networks) if active_network_idx is None else active_network_idx

    X_train = []
    for id in range(n_networks):
        if id in active_network_idx:
            X = []
            for i in range(1, n_clients, 2):
                x, _ = make_swiss_roll(n_samples=T, noise=noise)
                # Make two-dimensional to easen visualization
                x = x[:, [0, 2]]

                x = (x - x.mean()) / x.std()

                X.append(x)

            X = np.concatenate(X, axis = -1)[..., :n_clients]
            X_train.append(torch.from_numpy(X).to(dtype=torch.float32))
        else:
            X_train.append(torch.empty((1, n_clients), dtype = torch.float32))

    if logger is not None:
        logger(f'Dataset size = {len(X_train)}')
        logger(f'Data shape = {X_train[0].shape}')


    if save_path is not None and plot_networks_list is not None and len(plot_networks_list):
        for network_id in plot_networks_list:
            fig, ax = plt.subplots(1, 1, figsize = (6, 6))
            ax.scatter(X_train[network_id][:, 0], X_train[network_id][:, 1], marker="*", alpha=0.2, label = 'Original data')
            # ax.set_title(f'Network #{network_id}')
            ax.legend(loc = 'best')

            plt.savefig(f'{save_path}/swissroll_data_network_{network_id}.pdf', dpi = 300)
            plt.close(fig)

    return X_train


def load_hypercube_dataset(data_shape = None, save_path = None, plot_networks_list = [], logger = None, **kwargs):
    '''
    Generate samples from a hypercube.
    '''
    shift = kwargs.get('shift', 0)
    scale = kwargs.get('scale', 1)
    active_network_idx = kwargs.get('active_network_idx', None)

    n_networks, n_clients, T = data_shape
    active_network_idx = range(n_networks) if active_network_idx is None else active_network_idx

    standardize = False
    X = shift + scale * (2 * torch.rand(size=(n_networks, n_clients, T), dtype = torch.float32) - 1)

    X_train = []
    for id in range(n_networks):
        if id in active_network_idx:
            x = torch.cat([X[id, client].reshape(-1, 1) for client in range(n_clients)], dim = -1)
            x = (x - x.mean()) / x.std() if standardize else x
            X_train.append(x)
        else:
            X_train.append(torch.empty((1, n_clients), dtype = torch.float32))

    if logger is not None:
        logger(f'Dataset size = {len(X_train)}')
        logger(f'Data shape = {X_train[0].shape}')


    if save_path is not None and plot_networks_list is not None and len(plot_networks_list):
        for network_id in plot_networks_list:
            fig, ax = plt.subplots(1, 1, figsize = (6, 6))
            ax.scatter(X_train[network_id][:, 0], X_train[network_id][:, 1], marker="*", alpha=0.2, label = 'Original data')
            # ax.set_title(f'Network #{network_id}')
            ax.legend(loc = 'best')

            plt.savefig(f'{save_path}/hypercube_data_network_{network_id}.pdf', dpi = 300)
            plt.close(fig)

    return X_train


def make_loss_fn(eval_type):
    if eval_type == 'l1':
        loss_fn = torch.nn.L1Loss(reduction='none')
    elif eval_type == 'l2':
        loss_fn = torch.nn.MSELoss(reduction='none')
    elif eval_type == 'huber':
        loss_fn = torch.nn.SmoothL1Loss(reduction='none')
    else:
        raise NotImplementedError

    def weighted_loss_fn(input, target, weights = None):
        weights = 1. if weights is None else weights
        return torch.mean(weights * loss_fn(input, target))
    
    return weighted_loss_fn
    

def load_dataset(args, logger):

    if args.dataset == 'swiss-roll':
        # Load swissroll dataset
        X_train = load_swissroll_dataset(data_shape=(args.n_networks, args.n, args.N),
                                         save_path = f'{args.root}/results/{args.save_dir}/plots',
                                         plot_networks_list=args.plot_networks_list,
                                         logger=logger,
                                         noise = 0.1, # 0.5
                                         active_network_idx = [args.train_single_network_id] if args.train_single_network_id is not None else None
                                         )

    elif args.dataset == 'RRM':
        # Load RRM dataset obtained by training a state-augmented RRM policy
        X_train, Ps = load_RRM_dataset(dataset_path = f"{args.RRM_config_and_dataset_root_path}/{args.RRM_experiment_name}/dataset.pt",
                                        save_path=f'{args.root}/results/{args.save_dir}/plots',
                                        plot_networks_list=args.plot_networks_list,
                                        logger = logger,
                                        data_fabrication_factor=args.RRM_data_fabrication_factor,
                                        standardize=args.RRM_data_standardization)
        
    elif args.dataset == 'hypercube':
        # Load hypercube dataset
        X_train = load_hypercube_dataset(data_shape = (args.n_networks, args.n, args.N),
                                         save_path = f'{args.root}/results/{args.save_dir}/plots',
                                         plot_networks_list=args.plot_networks_list,
                                         logger = logger,
                                         shift = 0.5,
                                         scale = 0.1,
                                         active_network_idx = [args.train_single_network_id] if args.train_single_network_id is not None else None
                                         )
    else:
        raise NotImplementedError

    return X_train
    


  
# def create_model(args, device = 'cpu'):
#     '''
#     Create and initialize the Diffusion Model.
#     '''

#     if args.model_architecture == 'GNN':
#         nunits = args.model_nunits if hasattr(args, 'model_nunits') and args.model_nunits is not None else 64
#         nblocks = args.model_nlayers if hasattr(args, 'model_nlayers') and args.model_nlayers is not None else 4
#         nonlinearity = args.model_nonlinearity if hasattr(args, 'model_nonlinearity') and args.model_nonlinearity is not None else 'relu'
#         batch_norm = args.model_batch_norm if hasattr(args, 'model_batch_norm') and args.model_batch_norm is not None else False

#         model = GraphDiffusionModel(nfeatures=1, nblocks=nblocks, nunits=nunits, nonlinearity=nonlinearity, batch_norm=batch_norm)
    
#     elif args.model_architecture == 'FCNN':
#         nblocks = args.model_nlayers if hasattr(args, 'model_nlayers') and args.model_nlayers is not None else 4 # 2
#         nunits = args.model_nunits if hasattr(args, 'model_nunits') and args.model_nunits is not None else 64 # 64
#         model = DiffusionModel(nfeatures=args.n, nblocks=nblocks, nunits = nunits)

#     elif args.model_architecture == 'ConditionalLinear':
#         nblocks = args.model_nlayers if hasattr(args, 'model_nlayers') and args.model_nlayers is not None else 4 # 2
#         nunits = args.model_nunits if hasattr(args, 'model_nunits') and args.model_nunits is not None else 64 # 64
#         model = ConditionalLinearModel(nsteps=args.diffusion_steps, nfeatures=args.n, nblocks=nblocks, nunits=nunits)

#     else:
#         raise NotImplementedError
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)

#     if hasattr(args, 'load_model_chkpt_path') and args.load_model_chkpt_path is not None:
#         try:
#             model.load_state_dict(torch.load(args.load_model_chkpt_path))
#             print('Loading pretrained model weights is successful.')
#         except:
#             print(f'Loading model state dict from path {args.load_model_chkpt_path} failed.\nDefault weight initialization is applied.')

#     return model


# projecting RRM decisions to [0, 1] range
def projection(X, min = 0, max = 1., method = 'clamp'):
    if method == 'clamp':
        X_projected = torch.clamp(X, min=min, max=max)
    else:
        raise NotImplementedError

    return X_projected


def fabricate_RRM_data(Ps_orig, nsamples = None, method = 'perturbation', perturbation_sigma = 0.05, combine_orig_and_fabricated_data = False):
    '''
    This function fabricates new RRM trajectory data.
    
    Inputs:
        Ps_orig: A numpy array of RRM decisions with shape (n_networks, n_clients, n_samples_orig).
        nsamples: Desired number of new fabricated data
        method: Choice of data fabrication method.
        perturbation_sigma: Noise standard deviation if data fabrication method is perturbation.
        combine_orig_and_fabricated_data: If True, fabricated data is combined with original data.

    Outputs:
        Ps: A numpy array of fabricated RRM decisions with shape (n_networks, n_clients, x).
        x = n_samples_orig + nsamples if combine_orig_and_fabricated_data = True, nsamples.
    '''
    nsamples = n_samples_orig if nsamples is None else nsamples


    n_networks, n_clients, n_samples_orig = Ps_orig.shape

    Ps = []
    for network_id in range(n_networks):
        ps_orig = Ps_orig[network_id]
        sample_idx = torch.randint(low = 0, high=n_samples_orig, size = (nsamples,))
        ps = ps_orig[:, sample_idx]
        if method == 'perturbation': # add gaussian noise
            ps = ps + perturbation_sigma * torch.randn_like(ps)
            ps = projection(ps, min=0, max=1, method='clamp')
        else:
            raise NotImplementedError
        
        Ps.append(ps)

    Ps = torch.stack(Ps, dim = 0)
    if combine_orig_and_fabricated_data:
        Ps = torch.cat((Ps_orig, Ps), dim = -1)

    return Ps






def make_test_configs(args, algos = ['ITLinQ', 'FR', 'state-augmented-zeros', 'state-augmented-dual-dynamics', 'state-augmented-optimal-regression', 'state-augmented-dual-regression'],
                      lr_dual_list = None):
    test_configs = []

    ######################### lr_dual = 1.0 ########################
    # test_config = copy.deepcopy(args)
    # test_config.name = 'train_data_zeros_init_lr_dual_1.0'
    # test_config.test_on_train_data = True
    # test_config.dual_test_init_strategy = 'zeros'
    # test_config.lr_dual = 1.0
    # test_configs.append(test_config)
    

    # test_config = copy.deepcopy(args)
    # test_config.name = 'train_data_greedy_init_lr_dual_1.0'
    # test_config.test_on_train_data = True
    # test_config.dual_test_init_strategy = 'optimal-regression'
    # test_config.lr_dual = 1.0
    # test_configs.append(test_config)


     ######################### lr_dual = 0.1 ########################
    # test_config = copy.deepcopy(args)
    # test_config.name = 'train_data_zeros_init_lr_dual_0.1'
    # test_config.test_on_train_data = True
    # test_config.dual_test_init_strategy = 'zeros'
    # test_config.lr_dual = 0.1
    # test_config.dual_load_path = None
    # test_configs.append(test_config)


    # test_config = copy.deepcopy(args)
    # test_config.name = 'train_data_greedy_init_lr_dual_0.1'
    # test_config.test_on_train_data = True
    # test_config.dual_test_init_strategy = 'optimal-regression'
    # test_config.lr_dual = 0.1
    # test_configs.append(test_config)

    # test_config = copy.deepcopy(args)
    # test_config.name = 'train_data_mean_init_lr_dual_0.1'
    # test_config.test_on_train_data = True
    # test_config.dual_test_init_strategy = 'mean-regression'
    # test_config.lr_dual = 0.1
    # test_configs.append(test_config)


     ######################### lr_dual = 0.01 ########################
     
    # test_config = copy.deepcopy(args)
    # test_config.name = 'train_data_zeros_init_lr_dual_0.001'
    # test_config.test_on_train_data = True
    # test_config.dual_test_init_strategy = 'zeros'
    # test_config.lr_dual = 0.001
    # test_configs.append(test_config)

    # test_config = copy.deepcopy(args)
    # test_config.name = 'train_data_greedy_init_lr_dual_0.001'
    # test_config.test_on_train_data = True
    # test_config.dual_test_init_strategy = 'optimal-regression'
    # test_config.lr_dual = 0.001
    # test_config.dual_load_path = None
    # test_configs.append(test_config)
    
    if 'ITLinQ' in algos:
        # ITLinQ baseline #
        test_config = copy.deepcopy(args)
        test_config.alg = 'ITLinQ'
        test_config.name = test_config.alg
        test_configs.append(test_config)
    
    if 'FR' in algos:
        # FR baseline #
        test_config = copy.deepcopy(args)
        test_config.alg = 'FR'
        test_config.name = test_config.alg
        test_configs.append(test_config)
    
    lr_dual_list = [args.lr_dual['test']] if lr_dual_list is None or len(lr_dual_list) == 0 else lr_dual_list

    for lr_dual in lr_dual_list:
        TEST_ON_TRAIN_DATA = args.test_on_train_data
        if 'state-augmented-zeros' in algos:
            test_config = copy.deepcopy(args)
            test_config.test_on_train_data = TEST_ON_TRAIN_DATA
            test_config.dual_test_init_strategy = 'zeros'
            test_config.lr_dual = lr_dual
            test_config.lr_resilient = args.lr_resilient['test']
            test_config.name = f'state-augmented-zeros-init'
            test_configs.append(test_config)


        if TEST_ON_TRAIN_DATA and 'state-augmented-dual-dynamics' in algos:
            test_config = copy.deepcopy(args)
            test_config.test_on_train_data = TEST_ON_TRAIN_DATA
            test_config.dual_test_init_strategy = 'dual-dynamics'
            test_config.lr_dual = lr_dual
            test_config.lr_resilient = args.lr_resilient['test']
            test_config.dual_load_path = None
            test_config.name = f'state-augmented-dual-dynamics-init'
            test_configs.append(test_config)


        if TEST_ON_TRAIN_DATA and 'state-augmented-optimal-regression' in algos:
            test_config = copy.deepcopy(args)
            test_config.test_on_train_data = TEST_ON_TRAIN_DATA
            test_config.dual_test_init_strategy = 'optimal-regression'
            test_config.lr_dual = lr_dual
            test_config.lr_resilient = args.lr_resilient['test']
            test_config.dual_load_path = None
            test_config.name = f'state-augmented-optimal-regression'
            test_configs.append(test_config)


        if 'state-augmented-dual-regression' in algos:
            test_config = copy.deepcopy(args)
            test_config.test_on_train_data = TEST_ON_TRAIN_DATA
            test_config.dual_test_init_strategy = 'dual-regression'
            test_config.lr_dual = lr_dual
            test_config.lr_resilient = args.lr_resilient['test']
            test_config.dual_load_path = None
            test_config.name = f'state-augmented-dual-regression'
            test_configs.append(test_config)

    return test_configs
    

def seed_everything(seed):
    # set the random seed
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_substring_index(string, substring):
    return string.index(substring) + len(substring) - 1


def make_experiment_name(args):
    # Create experiment name
    experiment_name = 'm_{}_T_{}_r_min_{}_train_{}_test_{}_mode_{}_seed_{}'.format(args.m,
                                                                            max(args.T['train'], args.T['test']),
                                                                            args.r_min,
                                                                            args.num_samples['train'],
                                                                            args.num_samples['test'],
                                                                            args.density_mode,
                                                                            args.random_seed
                                                                            )
    
    if hasattr(args, 'R'):
        if abs(args.R - 2000) > 1e-6 and args.density_mode == 'var_density':
            experiment_name += f'_R_{args.R}'
    if hasattr(args, 'density_n_users_per_km_squared'):
        if abs(args.density_n_users_per_km_squared - 5.) > 1e-6 and args.density_mode == 'fixed_density':
            experiment_name += f'_user_density_{args.density_n_users_per_km_squared}'
    if hasattr(args, 'shadowing'):
        if abs(args.shadowing - 7) > 1e-6: # default value 7
            experiment_name += f'_shadowing_{args.shadowing}'
    if hasattr(args, 'speed'):
        if abs(args.speed - 1.0) > 1e-6: # default value 1.0
            experiment_name += f'_speed_{args.speed}'
    if hasattr(args, 'num_fading_paths'):
        if abs(args.num_fading_paths - 100) > 1e-6: # default value 100
            experiment_name += f'_num_fading_paths_{args.num_fading_paths}'

    channel_data_glob_path = experiment_name
    r_min_end_idx = find_substring_index(channel_data_glob_path, 'r_min_')
    train_start_idx = find_substring_index(channel_data_glob_path, '_train') - len('_train') + 1
    channel_data_glob_path = channel_data_glob_path[:r_min_end_idx+1] + '*' + channel_data_glob_path[train_start_idx:]

    return experiment_name, channel_data_glob_path


def make_dual_transform_fnc(eval_type):
    if eval_type == 'softmax':
        def dual_transform(mu):
            return torch.nn.functional.softmax(mu, dim = -1)
    elif eval_type == 'L2_normalization':
        def dual_transform(mu):
            return torch.nn.functional.normalize(mu, p=2, dim = -1)
    elif eval_type == 'none' or eval_type is None:
        def dual_transform(mu):
            return mu
    else:
        raise NotImplementedError
    
    return dual_transform


def compute_hist_bins(a, axis = None, **kwargs):
    nbins = kwargs.get('nbins', 10)
    binWidth = kwargs.get('binWidth', 0.1)

    bins = 0 + np.arange(nbins + 1) * binWidth # np.linspace(start=0, retstep=binWidth, num=nbins + 1).tolist()
    hist = []
    for i in range(len(bins) - 1):
        temp_hist = np.mean((a >= bins[i]) * (a < bins[i+1]), axis = axis)
        hist.append(temp_hist)
    hist = np.stack(hist, axis=axis)
    # hist = hist * 1/hist.sum(axis=axis, keepdims = True)
    # hist, binEdges = np.histogram(a=p, bins=bins, density=False, axis = axis, weights=np.ones_like(p) / )
    return hist


# calculating SINR features
def calc_log_SINR(p, gamma, h, noise_var, method = 'log_sinr'):
    """
    calculate rates for a batch of b networks, each with m transmitters and n recievers
    inputs:
        p: bm x 1 tensor containing transmit power levels
        gamma: bn x 1 tensor containing user scheduling decisions
        h: b x (m+n) x (m+n) weighted adjacency matrix containing instantaneous channel gains
        noise_var: scalar indicating noise variance
        metric: Four different choices
            - log_sinr: calculate log sinr
            - log snr: calculate log snr
            - log sir: calculate log sir
            - log sI: calculate log sI = log (1 + S / (max I))
    output:
        sinrs: bn x 1 tensor containing user sinrs
    """
    b = h.shape[0]
    p = p.view(b, -1, 1)
    gamma = gamma.view(b, -1, 1)
    m = p.shape[1]
    
    combined_p_gamma = torch.bmm(p, torch.transpose(gamma, 1, 2))
    signal = torch.sum(combined_p_gamma * h[:, :m, m:], dim=1)
    
    if method == 'log_sinr':
        interference = torch.sum(combined_p_gamma * torch.transpose(h[:, m:, :m], 1, 2), dim=1)
        rates = torch.log2(signal / (noise_var + interference)).view(-1, 1)
    elif method == 'log_snr':
        rates = torch.log2(signal / noise_var).view(-1, 1)
    elif method == 'log_sir':
        interference = torch.sum(combined_p_gamma * torch.transpose(h[:, m:, :m], 1, 2), dim=1)
        rates = torch.log2(signal / (interference)).view(-1, 1)
    elif method == 'log_sI':
        interference = torch.amax(combined_p_gamma * torch.transpose(h[:, m:, :m], 1, 2), dim=1)
        rates = torch.log2(signal / (interference)).view(-1, 1)
    
    return rates


def threshold_multipliers(lambdas, thresh_low = 0.0, thresh_high = None):
    '''
    Filter the network and client indices corresponding to dual multipliers within
    [thresh_low, thresh_high].
    '''

    n_graphs, n, n_mixtures = lambdas.shape
    if thresh_high is None:
        temp_graph_idx, temp_client_idx = np.where(lambdas[:, :, 0] >= thresh_low)
    else:
        temp_graph_idx, temp_client_idx = np.where((lambdas[:, :, 0] >= thresh_low) * (lambdas[:, :, 0] <= thresh_high))

    graph_idx = np.unique(temp_graph_idx)
    client_idx = [[] for _ in range(len(graph_idx))]
    for idx, graph in enumerate(graph_idx):
        client_idx[idx] = temp_client_idx[np.where(temp_graph_idx == graph_idx[idx])].tolist()

    return graph_idx, client_idx


def sample_n_simplex(n):
  ''' Return uniformly random vector in the n-simplex '''
  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)


# calculating rates
def calc_rates(p, gamma, h, noise_var):
    """
    calculate rates for a batch of b networks, each with m transmitters and n recievers
    inputs:
        p: bm x 1 tensor containing transmit power levels
        gamma: bn x 1 tensor containing user scheduling decisions
        h: b x (m+n) x (m+n) weighted adjacency matrix containing instantaneous channel gains
        noise_var: scalar indicating noise variance
        training: boolean variable indicating whether the models are being trained or not; during evaluation, 
        entries of gamma are forced to be integers to satisfy hard user scheudling constraints
        
    output:
        rates: bn x 1 tensor containing user rates
    """
    b = h.shape[0]
    p = p.view(b, -1, 1)
    gamma = gamma.view(b, -1, 1)
    m = p.shape[1]
    
    combined_p_gamma = torch.bmm(p, torch.transpose(gamma, 1, 2))
    signal = torch.sum(combined_p_gamma * h[:, :m, m:], dim=1)
    interference = torch.sum(combined_p_gamma * torch.transpose(h[:, m:, :m], 1, 2), dim=1)
    
    rates = torch.log2(1 + signal / (noise_var + interference)).view(-1, 1)
    
    return rates

# baseline ITLinQ method
def ITLinQ(H_raw, Pmax, noise_var, PFs):
    H = H_raw * Pmax / noise_var
    n = np.shape(H)[0]
    prity = np.argsort(PFs)[-1:-n-1:-1]
    flags = np.zeros(n)
    M = 10 ** 2.5
    eta = 0.5
    flags[prity[0]] = 1
    for pair in prity[1:]:
        SNR = H[pair,pair]
        INRs_in = [H[TP,pair] for TP in range(n) if flags[TP]]
        INRs_out = [H[pair,UE] for UE in range(n) if flags[UE]]
        max_INR_in = max(INRs_in)
        max_INR_out = max(INRs_out)
        if max(max_INR_in,max_INR_out) <= M * (SNR ** eta):
            flags[pair] = 1
    return flags * Pmax

def convert_channels(a, snr):
    a_flattened = a[a > 0]
    a_flattened_log = np.log(snr * a_flattened)
    a_norm = LA.norm(a_flattened_log)
    a_log = np.log(snr * a)
    a_log[a == 0] = 0
    return a_log / a_norm

class Data_modTxIndex(Data):
    def __init__(self,
                 y=None,
                 edge_index_l=None,
                 edge_weight_l=None,
                 edge_index=None,
                 edge_weight=None,
                 weighted_adjacency=None,
                 weighted_adjacency_l=None,
                 transmitters_index=None,
                 init_long_term_avg_rates=None,
                 num_nodes=None,
                 m=None):
        super().__init__()
        self.y = y
        self.edge_index_l = edge_index_l
        self.edge_weight_l = edge_weight_l
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.weighted_adjacency = weighted_adjacency
        self.weighted_adjacency_l = weighted_adjacency_l
        self.transmitters_index = transmitters_index
        self.init_long_term_avg_rates = init_long_term_avg_rates
        self.num_nodes = num_nodes
        self.m = m
                
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'transmitters_index':
            return self.m
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
class WirelessDataset(Dataset):
    def __init__(self, data_list):
        super().__init__(None, None, None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx], idx