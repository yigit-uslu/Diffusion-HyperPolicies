import abc
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import seaborn as sns
from torch_scatter import scatter

from core.config import default_log_freq_rl, default_log_freq_dm


plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}',

})

plt.style.use('bmh')



class Logger(abc.ABC):
    def __init__(self, data, log_metric, log_path):
        self.data = data
        self.log_metric = log_metric
        self.log_path = log_path

    # @abc.abstractmethod
    def update_data(self, new_data):
        self.data.append(new_data)

    def reset_data(self):
        self.data = []

    @abc.abstractmethod
    def __call__(self):
        pass

    # @abc.abstractmethod
    def get_save_data(self):
        return {'data': self.data,
                'log_metric': self.log_metric,
                'log_path': self.log_path
                }

    @property
    def log_freq(self):
        return default_log_freq_rl
    
    # @log_freq.setter
    # def loq_freq(self, new_log_freq):
    #     if new_log_freq > 0 and isinstance(new_log_freq, int):
    #         self.log_freq = new_log_freq
    #     else:
    #         raise ValueError("Log freq should be a positive integer!")
    


class RLTrainLossLogger(Logger):
    def __init__(self, data, log_path):
        super(RLTrainLossLogger, self).__init__(data=data, log_metric='loss', log_path=log_path)

    def update_data(self, new_data):
        if self.log_metric not in new_data:
            return
        
        if 'epoch' in new_data:
            epoch = new_data['epoch']
        
        data = new_data[self.log_metric].mean().item()
        super().update_data( (epoch, data) )

    def __call__(self):

        if not len(self.data):
            return
        
        try:
            epochs, L = zip(*self.data)

            epochs_tensor = torch.tensor(epochs)
            L_tensor = torch.tensor(L)

            L = scatter(src=L_tensor, dim=0, index=epochs_tensor, reduce="mean").numpy()
            epochs = np.arange(len(L))

            # print("L_tensor.shape: ", L_tensor.shape)
            # print("L.shape: ", L.shape)

        except:
            L = np.array(self.data)
            epochs = np.arange(len(L))


        if self.log_path is not None and (epochs[-1] + 1) % self.log_freq == 0: # and epochs.max().item() + 1 % self.log_freq == 0:

            fig, ax = plt.subplots(1, 1, figsize = (8, 4))
            ax.plot(epochs, L, '-r')
            ax.set_xlabel('Epoch (n)')
            ax.set_ylabel(self.log_metric)
            ax.grid(True)
            fig.tight_layout()

            os.makedirs(f"{self.log_path}", exist_ok=True)
            plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)
            plt.close(fig)



class DiffusionTrainLossLogger(Logger):
    def __init__(self, data, log_path):
        super(DiffusionTrainLossLogger, self).__init__(data=data, log_metric='loss', log_path=log_path)

    def update_data(self, new_data):
        if self.log_metric not in new_data:
            return
        
        if 'epoch' in new_data:
            epoch = new_data['epoch']
        
        data = new_data[self.log_metric].mean().item()
        super().update_data( (epoch, data) )

    @property
    def log_freq(self):
        return default_log_freq_dm
    
    # @log_freq.setter
    # def loq_freq(self, new_log_freq):
    #     if new_log_freq > 0 and isinstance(new_log_freq, int):
    #         self.log_freq = new_log_freq
    #     else:
    #         raise ValueError("Log freq should be a positive integer!")

    def __call__(self):

        if not len(self.data):
            return
        
        try:
            epochs, L = zip(*self.data)

            epochs_tensor = torch.tensor(epochs)
            L_tensor = torch.tensor(L)

            L = scatter(src=L_tensor, dim=0, index=epochs_tensor, reduce="mean").numpy()
            epochs = np.arange(len(L))

            # print("L_tensor.shape: ", L_tensor.shape)
            # print("L.shape: ", L.shape)

        except:
            L = np.array(self.data)
            epochs = np.arange(len(L))


        if self.log_path is not None and (epochs[-1] + 1) % self.log_freq == 0: # and epochs.max().item() + 1 % self.log_freq == 0:

            fig, ax = plt.subplots(1, 1, figsize = (8, 4))
            ax.plot(epochs, L, '-r')
            ax.set_xlabel('Epoch (n)')
            ax.set_ylabel(self.log_metric)
            ax.grid(True)
            fig.tight_layout()

            os.makedirs(f"{self.log_path}", exist_ok=True)
            plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)
            plt.close(fig)



class LambdaScatterLogger(Logger):
    def __init__(self, data, log_path):
        super(LambdaScatterLogger, self).__init__(data=data, log_metric='scatter-lambdas', log_path=log_path)

    def update_data(self, new_data):
        self.reset_data() # do not concatenate results along epochs

        if self.log_metric not in new_data:
            return
        
        if 'epoch' in new_data:
            epoch = new_data['epoch']
        
        data = new_data[self.log_metric]
        super().update_data( (epoch, data) )

    @property
    def log_freq(self):
        return default_log_freq_dm
    
    # @log_freq.setter
    # def loq_freq(self, new_log_freq):
    #     if new_log_freq > 0 and isinstance(new_log_freq, int):
    #         self.log_freq = new_log_freq
    #     else:
    #         raise ValueError("Log freq should be a positive integer!")
        

    def __call__(self, lambdas_label = [None, None]):

        if not len(self.data):
            return
        
        epoch, lambdas = zip(*self.data)
        epoch = epoch[0]
        lambdas = lambdas[0]

        if isinstance(lambdas, tuple):
            lambdas, lambdas_orig = lambdas
        else:
            lambdas_orig = None

        df_lambdas = pd.DataFrame(lambdas, columns = [r"$\lambda_1$", r"$\lambda_2$"]) if lambdas is not None else None
        df_lambdas_orig = pd.DataFrame(lambdas_orig, columns = [r"$\lambda_1$", r"$\lambda_2$"]) if lambdas_orig is not None else None


        if self.log_path is not None and (epoch + 1) % self.log_freq == 0: # and epochs.max().item() + 1 % self.log_freq == 0:

            fig, ax = plt.subplots(1, 1, figsize = (8, 4))

            if df_lambdas is not None:
                sns.scatterplot(x=df_lambdas[r"$\lambda_1$"], y=df_lambdas[r"$\lambda_2$"], marker = '1', alpha = 0.5, label = r"DDPM-sampled $\lambda$" if lambdas_label[0] is None else lambdas_label[0], ax = ax)

            if df_lambdas_orig is not None:
                sns.scatterplot(x=df_lambdas_orig[r"$\lambda_1$"], y=df_lambdas_orig[r"$\lambda_2$"], marker = "o", alpha = 0.5, label = r"Imp.-sampled $\lambda$" if lambdas_label[1] is None else lambdas_label[1], ax = ax)
           

            ax.set_ylabel(self.log_metric)
            ax.grid(True)
            fig.tight_layout()

            os.makedirs(f"{self.log_path}", exist_ok=True)
            plt.savefig(f'{self.log_path}/{self.log_metric}-epoch-{epoch}.pdf', dpi = 300)
            plt.close(fig)



class RLTrainPGradNormLogger(RLTrainLossLogger):
    def __init__(self, data, log_path):
        super().__init__(data, log_path)
        self.log_metric = 'pgrad_norm'


class DiffusionTrainPGradNormLogger(DiffusionTrainLossLogger):
    def __init__(self, data, log_path):
        super().__init__(data, log_path)
        self.log_metric = 'pgrad_norm'


class RLTrainAugRewardsLogger(RLTrainLossLogger):
    def __init__(self, data, log_path):
        super().__init__(data, log_path)
        self.log_metric = 'aug_rewards'


class RLTrainRewardsLogger(RLTrainLossLogger):
    def __init__(self, data, log_path):
        super().__init__(data, log_path)
        self.log_metric = 'rewards'

    def update_data(self, new_data):
        if self.log_metric not in new_data:
            return
        
        if 'epoch' in new_data:
            epoch = new_data['epoch']
        
        data = new_data[self.log_metric]
        Logger.update_data( self, (epoch, data) )

    def __call__(self):

        if not len(self.data):
            return
        try:
            epochs, L = zip(*self.data)

            epochs_tensor = torch.tensor(epochs)
            L_tensor = torch.tensor(L)

            L = scatter(src=L_tensor, dim=0, index=epochs_tensor, reduce="mean").numpy()
            epochs = np.arange(len(L))

            # print("L_tensor.shape: ", L_tensor.shape)
            # print("L.shape: ", L.shape)

        except:
            L = np.array(self.data)
            epochs = np.arange(len(L))

        if self.log_path is not None and (epochs[-1] + 1) % self.log_freq == 0: #and epochs.max().item() + 1 % self.log_freq == 0:

            fig, ax = plt.subplots(1, 1, figsize = (6, 6))
            ax.plot(epochs, L, label = [r"$r_{" + str(i) + '}$' for i in range(L.shape[-1])])
            ax.set_xlabel('Epoch (n)')
            ax.set_ylabel(self.log_metric)
            ax.legend(loc = 'best')
            ax.grid(True)
            fig.tight_layout()

            os.makedirs(f"{self.log_path}", exist_ok=True)
            plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)
            plt.close(fig)


class RLTestRewardsLogger(RLTrainLossLogger):
    def __init__(self, data, log_path):
        super().__init__(data, log_path)
        self.log_metric = 'rewards'

    def update_data(self, new_data):
        self.reset_data() # do not append data
        if self.log_metric not in new_data:
            return
        
        if 'epoch' in new_data:
            epoch = new_data['epoch']
        else:
            epoch = None
        
        data = new_data[self.log_metric]

        Logger.update_data(self, (epoch, data))


    def __call__(self):

        if not len(self.data):
            return
        
        try:
            epochs, Ls = zip(*self.data)

        except:
            Ls = np.array(self.data)
            # epochs = np.arange(len(L))

        if self.log_path is not None and (epochs[-1] is None or (epochs[-1] + 1) % self.log_freq == 0):

            for epoch, L in zip(epochs, Ls):
                fig, ax = plt.subplots(1, 1, figsize = (6, 6))
                for batch, l in enumerate(L):

                    n_timesteps, n_rewards = l.shape

                    ax.plot(np.arange(n_timesteps), l, label = [r"$r^{(" + str(batch) + ")}" + "_{" + str(i) + '}$' for i in range(n_rewards)])
                ax.set_xlabel('Timestep (t)')
                ax.set_ylabel(self.log_metric)
                ax.legend(loc = 'best')
                ax.grid(True)
                fig.tight_layout()

                os.makedirs(f"{self.log_path}", exist_ok=True)

                if epoch is not None:
                    plt.savefig(f'{self.log_path}/{self.log_metric}-epoch-{epoch}.pdf', dpi = 300)
                else:
                    plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)

                plt.close(fig)


                # Cumulative averages
                fig, ax = plt.subplots(1, 1, figsize = (6, 6))
                for batch, l in enumerate(L):

                    l = np.divide(np.cumsum(l, axis = 0), np.cumsum(np.ones_like(l), axis = 0))

                    n_timesteps, n_rewards = l.shape

                    ax.plot(np.arange(n_timesteps), l, label = [r"$r^{(" + str(batch) + ")}" + "_{" + str(i) + '}$' for i in range(n_rewards)])
                ax.set_xlabel('Timestep (t)')
                ax.set_ylabel('ergodic-' + self.log_metric)
                ax.legend(loc = 'best')
                ax.grid(True)
                fig.tight_layout()

                os.makedirs(f"{self.log_path}", exist_ok=True)

                if epoch is not None:
                    plt.savefig(f'{self.log_path}/ergodic-{self.log_metric}-epoch-{epoch}.pdf', dpi = 300)
                else:
                    plt.savefig(f'{self.log_path}/ergodic-{self.log_metric}.pdf', dpi = 300)

                plt.close(fig)



class RLTestLambdasLogger(RLTrainLossLogger):
    def __init__(self, data, log_path):
        super().__init__(data, log_path)
        self.log_metric = 'lambdas'

    def update_data(self, new_data):
        self.reset_data() # do not append data
        if self.log_metric not in new_data:
            return
        
        if 'epoch' in new_data:
            epoch = new_data['epoch']
        else:
            epoch = None
        
        data = new_data[self.log_metric]

        Logger.update_data(self, (epoch, data))


    def __call__(self):

        if not len(self.data):
            return
        
        try:
            epochs, Ls = zip(*self.data)

        except:
            Ls = np.array(self.data)
            # epochs = np.arange(len(L))

        if self.log_path is not None and (epochs[-1] is None or (epochs[-1] + 1) % self.log_freq == 0):

            for epoch, L in zip(epochs, Ls):
                fig, ax = plt.subplots(1, 1, figsize = (6, 6))
                for batch, l in enumerate(L):

                    # # Skip the current color in the default cycle by modifying rcParams
                    # # current_colors = plt.cm.tab10.colors
                    # current_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=current_colors[1:])  # Skip the first color

                    # Plot a dummy invisible line that won't appear in the legend
                    ax.plot([0, 1, 2], [0, 1, 2], linewidth = 0.0, label='_nolegend_')


                    n_timesteps, n_constraints = l.shape
                    ax.plot(np.arange(n_timesteps), l, label = [r"$\lambda^{(" + str(batch) + ")}" + "_{" + str(i) + '}$' for i in range(1, n_constraints+1)])

                ax.set_xlabel('Timestep (t)')
                ax.set_ylabel(self.log_metric)
                ax.legend(loc = 'best')
                ax.grid(True)
                fig.tight_layout()

                os.makedirs(f"{self.log_path}", exist_ok=True)

                if epoch is not None:
                    plt.savefig(f'{self.log_path}/{self.log_metric}-epoch-{epoch}.pdf', dpi = 300)
                else:
                    plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)

                plt.close(fig)


class RLTestLambdaScatterLogger(LambdaScatterLogger):
    def __init__(self, data, log_path):
        super().__init__(data=data, log_path=log_path)

    @property
    def log_freq(self):
        return default_log_freq_rl



class RLTestActionProbsLogger(RLTrainLossLogger):
    def __init__(self, data, log_path):
        super().__init__(data, log_path)
        self.log_metric = 'action-probs'

    def update_data(self, new_data):
        self.reset_data() # do not append data
        if self.log_metric not in new_data:
            return
        
        if 'epoch' in new_data:
            epoch = new_data['epoch']
        else:
            epoch = None
        
        data = new_data[self.log_metric]

        Logger.update_data(self, (epoch, data))


    def __call__(self):

        if not len(self.data):
            return
        
        try:
            epochs, Ls = zip(*self.data)

        except:
            Ls = np.array(self.data)
            # epochs = np.arange(len(L))

        if self.log_path is not None and (epochs[-1] is None or (epochs[-1] + 1) % self.log_freq == 0):

            for epoch, L in zip(epochs, Ls):
                fig, ax = plt.subplots(1, 1, figsize = (6, 6))
                for batch, l in enumerate(L):

                    n_timesteps, n_actions = l.shape

                    ax.plot(np.arange(n_timesteps), l, label = [r"$\pi_{" + str(i) + "}" + "(s_t, \lambda_t)^{(" + str(batch) + ")}$" for i in range(n_actions)])
                
                ax.set_xlabel('Timestep (t)')
                ax.set_ylabel(self.log_metric)
                ax.legend(loc = 'best')
                ax.grid(True)
                fig.tight_layout()

                os.makedirs(f"{self.log_path}", exist_ok=True)

                if epoch is not None:
                    plt.savefig(f'{self.log_path}/{self.log_metric}-epoch-{epoch}.pdf', dpi = 300)
                else:
                    plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)

                plt.close(fig)


class RLTestActionsLogger(RLTrainLossLogger):
    def __init__(self, data, log_path):
        super().__init__(data, log_path)
        self.log_metric = 'actions'

    def update_data(self, new_data):
        self.reset_data() # do not append data
        if self.log_metric not in new_data:
            return
        
        if 'epoch' in new_data:
            epoch = new_data['epoch']
        else:
            epoch = None
        
        data = new_data[self.log_metric]

        Logger.update_data(self, (epoch, data))


    def __call__(self):

        if not len(self.data):
            return
        
        try:
            epochs, Ls = zip(*self.data)

        except:
            Ls = np.array(self.data)
            # epochs = np.arange(len(L))

        if self.log_path is not None and (epochs[-1] is None or (epochs[-1] + 1) % self.log_freq == 0):

            for epoch, L in zip(epochs, Ls):
                fig, ax = plt.subplots(1, 1, figsize = (6, 6))
                for batch, l in enumerate(L):

                    n_timesteps = l.shape[0]

                    ax.plot(np.arange(n_timesteps), l, label = r"$a_t^{(" + str(batch) + ")}$")
                ax.set_xlabel('Timestep (t)')
                ax.set_ylabel(self.log_metric)
                ax.legend(loc = 'best')
                ax.grid(True)
                fig.tight_layout()

                os.makedirs(f"{self.log_path}", exist_ok=True)

                if epoch is not None:
                    plt.savefig(f'{self.log_path}/{self.log_metric}-epoch-{epoch}.pdf', dpi = 300)
                else:
                    plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)

                plt.close(fig)


class RLTestStatesLogger(RLTrainLossLogger):
    def __init__(self, data, log_path):
        super().__init__(data, log_path)
        self.log_metric = 'states'

    def update_data(self, new_data):
        self.reset_data() # do not append data
        if self.log_metric not in new_data:
            return
        
        if 'epoch' in new_data:
            epoch = new_data['epoch']
        else:
            epoch = None
        
        data = new_data[self.log_metric]

        Logger.update_data(self, (epoch, data))


    def __call__(self):

        if not len(self.data):
            return
        
        try:
            epochs, Ls = zip(*self.data)

        except:
            Ls = np.array(self.data)
            # epochs = np.arange(len(L))

        if self.log_path is not None and (epochs[-1] is None or (epochs[-1] + 1) % self.log_freq == 0):

            for epoch, L in zip(epochs, Ls):
                fig, ax = plt.subplots(1, 1, figsize = (6, 6))
                for batch, l in enumerate(L):

                    n_timesteps = l.shape[0]

                    ax.plot(np.arange(n_timesteps), l, label = r"$s_t^{(" + str(batch) + ")}$")
                ax.set_xlabel('Timestep (t)')
                ax.set_ylabel(self.log_metric)
                ax.legend(loc = 'best')
                ax.grid(True)
                fig.tight_layout()

                os.makedirs(f"{self.log_path}", exist_ok=True)

                if epoch is not None:
                    plt.savefig(f'{self.log_path}/{self.log_metric}-epoch-{epoch}.pdf', dpi = 300)
                else:
                    plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)

                plt.close(fig)


    


class RLQTableLogger(Logger):
    def __init__(self, data, log_path):
        super().__init__(data, log_path = log_path, log_metric="Q_values")
        # self.log_metric = 'Q_values'

    def update_data(self, new_data):
        self.reset_data()
        if self.log_metric not in new_data:
            return
        
        if 'epoch' in new_data:
            epoch = new_data['epoch']
        
        data = new_data[self.log_metric]
        super().update_data( (epoch, data) )

    # @property
    # def log_freq(self):
    #     return 50

    def __call__(self):

        if not len(self.data):
            return

        epoch, Q = zip(*self.data)
        epoch = epoch[0]
        Q = Q[0]

        if self.log_path is not None and (epoch + 1) % self.log_freq == 0: # and epoch + 1 % self.log_freq == 0:

            fig, ax = plt.subplots(1, 1, figsize = (6, 6))
            # ax.plot(epochs, L, '-r')

            Q = (Q - Q.min()) / (Q.max() - Q.min())

            if epoch == 2:
                print('Q: ', Q)

            # Create the heatmap using imshow()
            heatmap = ax.imshow(Q, cmap='viridis', interpolation='nearest')

            # Add color bar for scale
            plt.colorbar(heatmap)

            ax.set_ylabel('States')
            ax.set_xlabel('Actions')
            # ax.grid(True)
            fig.tight_layout()
            ax.set_title(f'Epoch {epoch}')

            os.makedirs(f"{self.log_path}", exist_ok=True)
            plt.savefig(f'{self.log_path}/{self.log_metric}-epoch-{epoch}.pdf', dpi = 300)
            plt.close(fig)



class LagrangiansImportanceSamplerLogger(Logger):
    def __init__(self, data, log_path):
        super(LagrangiansImportanceSamplerLogger, self).__init__(data=data, log_metric='importance-sampled-lagrangian', log_path=log_path)

    def update_data(self, new_data):
        if self.log_metric not in new_data:
            return
        
        if 'epoch' in new_data:
            epoch = new_data['epoch']
        
        data = new_data[self.log_metric].mean().item()
        super().update_data( (epoch, data) )

    def __call__(self):

        if not len(self.data):
            return
        try:
            epochs, L = zip(*self.data)

            epochs_tensor = torch.tensor(epochs)
            L_tensor = torch.tensor(L)

            L = scatter(src=L_tensor, dim=0, index=epochs_tensor, reduce="mean").numpy()
            epochs = np.arange(len(L))

            # print("L_tensor.shape: ", L_tensor.shape)
            # print("L.shape: ", L.shape)

        except:
            L = np.array(self.data)
            epochs = np.arange(len(L))


        if self.log_path is not None and (epochs[-1] + 1) % self.log_freq == 0: # and epochs.max().item() + 1 % self.log_freq == 0:

            fig, ax = plt.subplots(1, 1, figsize = (8, 4))
            ax.plot(epochs, L, '-r')
            ax.set_xlabel('Epoch (n)')
            ax.set_ylabel(self.log_metric)
            ax.grid(True)
            fig.tight_layout()

            os.makedirs(f"{self.log_path}", exist_ok=True)
            plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)
            plt.close(fig)



def make_rl_train_loggers(log_path):

    loggers = []
    logger = RLTrainLossLogger(data = [], log_path=log_path)
    loggers.append(logger)

    logger = RLTrainAugRewardsLogger(data = [], log_path = log_path)
    loggers.append(logger)

    logger = RLTrainRewardsLogger(data = [], log_path = log_path)
    loggers.append(logger)

    logger = RLTrainPGradNormLogger(data = [], log_path = log_path)
    loggers.append(logger)

    logger = RLQTableLogger(data = [], log_path = log_path)
    loggers.append(logger)

    return loggers


def make_dm_train_loggers(log_path):

    loggers = []

    logger = DiffusionTrainLossLogger(data = [], log_path=log_path)
    loggers.append(logger)

    logger = DiffusionTrainPGradNormLogger(data=[], log_path=log_path)
    loggers.append(logger)

    logger = LambdaScatterLogger(data = [], log_path=log_path)
    loggers.append(logger)

    return loggers

