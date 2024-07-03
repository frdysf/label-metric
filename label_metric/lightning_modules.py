import logging

import torch
import torch.nn as nn
import lightning as L

from label_metric.samplers import WeightManager
from label_metric.losses import TripletLoss

class LabelMetricModule(L.LightningModule):

    def __init__(
        self,
        backbone_model: nn.Module,
        prediction_head: nn.Module,
        my_logger: logging.Logger,
        weight_manager: WeightManager
    ):
        super().__init__()
        self.backbone_model = backbone_model
        self.prediction_head = prediction_head
        self.my_logger = my_logger
        self.weight_manager = weight_manager

    def forward(self, x: torch.Tensor):
        embeddings = self.backbone_model(x)
        return embeddings

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

if __name__ == '__main__':

    import lightning as L
    L.seed_everything(2024)
    from label_metric.utils.log_utils import setup_logger
    logger = logging.getLogger(__name__)
    setup_logger(logger)

    weight_manager = WeightManager(logger)

    from label_metric.data_modules import OrchideaSOLDataModule

    dm = OrchideaSOLDataModule(
        dataset_dir = '/data/scratch/acw751/_OrchideaSOL2020_release',
        min_num_per_leaf = 10,
        duration = 1.0,
        train_ratio = 0.8,
        valid_ratio = 0.1,
        logger = logger,
        more_level = 1,
        weight_manager = weight_manager,
        batch_size = 32, 
        num_workers = 2
    )

    dm.setup('fit')
    train_loader = dm.train_dataloader()
    valid_loader = dm.val_dataloader()

    training_batch = next(iter(train_loader))

    from label_metric.models import PlaceHolderModel, PredictionHead

    backbone_model = PlaceHolderModel(
        sr = 44100,
        n_fft = 2048,
        hop_length = 512,
        output_dim = 256
    )

    prediction_head = PredictionHead(
        embedding_size = 256,
        num_classes = len(dm.train_set.tree.leaves)
    )

    lightning_module = LabelMetricModule(
        backbone_model = backbone_model,
        prediction_head = prediction_head,
        my_logger = logger,
        weight_manager = weight_manager
    )

    lightning_module.training_step(training_batch, batch_idx=0)
