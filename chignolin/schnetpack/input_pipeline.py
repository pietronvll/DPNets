import logging

import torch


class TimeLaggedSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        lagtime: int = 1,
        shuffle: bool = False,
        drop_last: bool = True,
    ):
        self.num_points = len(dataset)
        self.batch_size = batch_size
        self.lagtime = lagtime
        self.shuffle = shuffle
        self.effective_sample = self.num_points - self.lagtime
        if not drop_last:
            logging.warning("drop_last = False is not supported.")
        self.drop_last = True

    def __iter__(self):
        if self.shuffle:
            idxs = torch.randperm(self.effective_sample)
        else:
            idxs = torch.arange(self.effective_sample)
        for i in range(self.effective_sample // self.batch_size):
            in_idxs = idxs[self.batch_size * i : self.batch_size * (i + 1)]
            out_idxs = in_idxs + self.lagtime
            batch_idxs = torch.zeros(self.batch_size * 2, dtype=in_idxs.dtype)
            batch_idxs[::2] = in_idxs
            batch_idxs[1::2] = out_idxs
            yield batch_idxs.tolist()

    def __len__(self):
        return self.effective_sample // self.batch_size


class StridedSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        lagtime: int = 1,
        shuffle: bool = False,
        drop_last: bool = True,
    ):
        self.num_points = len(dataset)
        self.batch_size = batch_size
        self.lagtime = lagtime
        if shuffle:
            logging.warning("shuffle = True is not supported.")
        self.shuffle = False
        self.effective_sample = self.num_points - self.lagtime
        if not drop_last:
            logging.warning("drop_last = False is not supported.")
        self.drop_last = True

    def __iter__(self):
        idxs = torch.arange(self.effective_sample)
        for i in range(self.effective_sample // self.batch_size):
            in_idxs = idxs[self.batch_size * i : self.batch_size * (i + 1)]
            out_idxs = in_idxs + self.lagtime
            batch_idxs = torch.zeros(self.batch_size * 2, dtype=in_idxs.dtype)
            batch_idxs[::2] = in_idxs
            batch_idxs[1::2] = out_idxs
            yield batch_idxs.tolist()

    def __len__(self):
        return self.effective_sample // self.batch_size
