import logging

from torch.utils.data import DataLoader
import lightning as L

from label_metric.datasets import BasicOrchideaSOL, TripletOrchideaSOL
from label_metric.samplers import TreeSampler
from label_metric.utils.log_utils import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)

class OrchideaSOLDataModule(L.LightningDataModule):
    
    def __init__(self,
        batch_size: int, 
        num_workers: int,
        more_level: int # for TreeSampler
        ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.more_level = more_level

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_set = TripletOrchideaSOL(split='train')
            self.tree_sampler = TreeSampler(self.train_set, self.more_level)
            self.valid_set = BasicOrchideaSOL(split='valid')
        if stage == 'test':
            self.test_set = BasicOrchideaSOL(split='test')

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            sampler = self.tree_sampler,
            batch_size=self.batch_size,
            num_workers = self.num_workers,
            drop_last = True,
        )

    def val_dataloader(self):
        return DataLoader()

    def test_dataloader(self):
        return DataLoader()
