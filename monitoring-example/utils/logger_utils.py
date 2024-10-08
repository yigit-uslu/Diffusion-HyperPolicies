import abc
import numpy as np
import torch
import matplotlib.pyplot as plt

class Logger(abc.ABC):
    def __init__(self, data, log_metric, log_path):
        self.data = data
        self.log_metric = log_metric
        self.log_path = log_path

    # @abc.abstractmethod
    def update_data(self, new_data):
        self.data.append(new_data)

    @abc.abstractmethod
    def __call__(self):
        pass


class DiffusionLossLogger(Logger):
    def __init__(self, data, log_path):
        super(DiffusionLossLogger, self).__init__(data=data, log_metric='loss', log_path=log_path)

    def update_data(self, new_data):
        if isinstance(new_data, np.ndarray) or isinstance(new_data, torch.Tensor):
            data = new_data[self.log_metric].mean().item()
        else:
            data = new_data[self.log_metric]
        super().update_data(data)

    def __call__(self):
        L = np.array(self.data)
        iters = np.arange(len(L))

        if self.log_path is not None:
            fig, ax = plt.subplots(1, 1, figsize = (8, 4))
            ax.plot(iters, L, '-r')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(self.log_metric)
            ax.grid(True)
            fig.tight_layout()

            plt.savefig(f'{self.log_path}/{self.log_metric}.pdf', dpi = 300)
            plt.close(fig)


class DiffusionPGradLogger(Logger):
    def __init__(self, data, log_path):
        super().__init__(data=data, log_metric='pgrad-norm', log_path=log_path)

    def update_data(self, new_data):
        return DiffusionLossLogger.update_data(new_data)

    def __call__(self):
        DiffusionLossLogger.__call__(self)