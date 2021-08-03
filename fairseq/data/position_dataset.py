# Author: Leo Zeyu Liu
# Date: July.25 2021
import torch

from fairseq.data import data_utils

from . import BaseWrapperDataset


class PositionDataset(BaseWrapperDataset):

    def __init__(self, dataset, pad_idx):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        # we add a offset to the calculated position bc nn.Embedding will turn pad_idx to be a 0 vector
        self.position_offset = self.pad_idx + 1

    def __getitem__(self, item):
        return self.position_offset + torch.arange(len(self.dataset[item]))

    def collater(self, samples):
        return data_utils.collate_tokens(samples, self.pad_idx)
