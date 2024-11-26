import torch
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        # Return the length of the shorter dataset
        return min([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        items = [dataset[idx] for dataset in self.datasets]
        return (*items,)


def importance_sampler(X_0, Y_0 = None, weights = None, batch_size = 1000, replacement = True):
    X_dataset = TensorDataset(X_0)
    if Y_0 is not None:
        Y_dataset = TensorDataset(Y_0)
        dataset = ZippedDataset([X_dataset, Y_dataset])
    else:
        dataset = X_dataset

    weights = torch.ones_like(len(X_dataset)) if weights is None else weights
    sampler = WeightedRandomSampler(weights, len(weights), replacement = replacement)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             sampler=sampler,
                                             batch_size = batch_size,
                                             drop_last = False
                                             )
    return dataloader