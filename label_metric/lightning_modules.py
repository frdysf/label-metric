import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

from label_metric.samplers import WeightManager
from label_metric.losses import TripletLoss

class LabelMetricModule(L.LightningModule):

    def __init__(
        self,
        backbone_model: nn.Module,
        prediction_head: nn.Module,
        triplet_loss_fn: TripletLoss,
        lambda_weight: float,
        my_logger: logging.Logger,
        weight_manager: WeightManager,
        learning_rate: float
    ):
        super().__init__()
        self.backbone_model = backbone_model
        self.prediction_head = prediction_head
        self.triplet_loss_fn = triplet_loss_fn
        assert 0 < lambda_weight < 1
        self.lambda_weight = torch.tensor(lambda_weight)
        self.my_logger = my_logger
        self.weight_manager = weight_manager
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor):
        embeddings = self.backbone_model(x)
        return embeddings

    def training_step(self, batch, batch_idx):
        epoch_idx = self.current_epoch
        # anchors, positives, negatives
        x_a, y_a = batch['anc']
        x_p, y_p = batch['pos']
        x_n, y_n = batch['neg']
        # class weights
        weights = self.weight_manager.get_weights()
        w_a = weights['anc'].to(self.device)
        w_p = weights['pos'].to(self.device)
        w_n = weights['neg'].to(self.device)
        # embeddings
        z_a = self(x_a)
        z_p = self(x_p)
        z_n = self(x_n)
        # triplet loss
        triplet_loss = self.triplet_loss_fn(
            anchor_embs = z_a, 
            positive_embs = z_p, 
            negative_embs = z_n
        ) * self.lambda_weight
        # classification loss
        logits_a = self.prediction_head(z_a)
        classification_loss = F.cross_entropy(
            input = logits_a, 
            target = y_a,
            weight = w_a
        ) * (1 - self.lambda_weight)
        loss = triplet_loss + classification_loss
        self.my_logger.info(f'training epoch {epoch_idx} batch {batch_idx} '
                            f'triplet loss: {triplet_loss} '
                            f'classification loss: {classification_loss}')
        return loss

    def validation_step(self, batch, batch_idx):
        epoch_idx = self.current_epoch
        x, y = batch
        logits = self.prediction_head(self(x))
        loss = F.cross_entropy(input = logits, target = y)
        self.my_logger.info(f'validation epoch {epoch_idx} batch {batch_idx} '
                            f'classification loss: {loss}')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__ == '__main__':

    # example code

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

    from label_metric.models import ConvModel, PredictionHead

    backbone_model = ConvModel(
        duration = 1.0,
        conv_out_channels = 128,
        embedding_size = 256,
        sr = 44100,
        n_fft = 2048,
        hop_length = 512,
    )

    prediction_head = PredictionHead(
        embedding_size = 256,
        num_classes = len(dm.train_set.tree.leaves)
    )

    from pytorch_metric_learning.distances import CosineSimilarity

    triplet_loss_fn = TripletLoss(margin=0.2, distance=CosineSimilarity())

    lightning_module = LabelMetricModule(
        backbone_model = backbone_model,
        prediction_head = prediction_head,
        triplet_loss_fn = triplet_loss_fn,
        lambda_weight = 0.95,
        my_logger = logger,
        weight_manager = weight_manager,
        learning_rate = 0.001
    )

    # for i, batch in enumerate(train_loader):
    #     lightning_module.training_step(batch, batch_idx=i)

    # for i, batch in enumerate(valid_loader):
    #     lightning_module.validation_step(batch, batch_idx=i)

    trainer = L.Trainer(max_epochs=5)
    trainer.fit(model = lightning_module, datamodule = dm)
