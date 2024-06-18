import logging

from torch.utils.data import DataLoader
import lightning as L

from label_metric.datasets import OrchideaSOL
from label_metric.samplers import TreeSampler
from label_metric.utils.log_utils import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)

class OrchideaSOLDataModule(L.LightningDataModule):
    
    def __init__(self, 
                 batch_size, 
                 num_workers,
                 more_level):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.more_level = more_level # should be in train_dataset_args

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_set = OrchideaSOL(split='train')
            self.valid_set = OrchideaSOL(split='valid')
        if stage == 'test':
            self.test_set = OrchideaSOL(split='test')

    def train_dataloader(self):
        sampler = TreeSampler(self.train_set, self.more_level)
        return DataLoader(self.train_set,
                          sampler = sampler,
                          batch_size=self.batch_size,
                          num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader()

    def test_dataloader(self):
        return DataLoader()
