import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torchmetrics import Accuracy
from torchmetrics.retrieval import RetrievalPrecision
from torchmetrics.functional.retrieval import retrieval_precision

from label_metric.samplers import WeightManager
from label_metric.losses import TripletLoss

class LabelMetricModule(L.LightningModule):

    def __init__(
        self,
        backbone_model: nn.Module,
        prediction_head_leaf: Optional[nn.Module],
        prediction_head_all_nodes_except_root: Optional[nn.Module],
        triplet_loss: TripletLoss,
        alpha: float,
        beta: float,
        my_logger: logging.Logger,
        weight_manager: WeightManager,
        learning_rate: float,
        classification_accuracy_top_k: int,
        retrieval_precision_top_k: int
    ):
        super().__init__()
        
        # models
        self.backbone_model = backbone_model
        self.prediction_head_leaf = prediction_head_leaf
        self.prediction_head_all_nodes_except_root = prediction_head_all_nodes_except_root
        
        # losses
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.binary_cross_entropy_loss = nn.BCEWithLogitsLoss()
        self.triplet_loss = triplet_loss
        
        # loss weights
        assert alpha >= 0 and beta >= 0
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        
        # class weight manager
        self.weight_manager = weight_manager
        
        # optimization
        self.learning_rate = learning_rate
        
        # evaluation
        self.ca_top_k = classification_accuracy_top_k
        self.rp_top_k = retrieval_precision_top_k
        self.accuracy = Accuracy(
            task = 'multiclass', 
            num_classes = prediction_head_leaf.num_classes
        )
        self.accuracy_top_k = Accuracy(
            task = 'multiclass', 
            num_classes = prediction_head_leaf.num_classes,
            top_k = self.ca_top_k
        )
        self.retrieval_precision = RetrievalPrecision(top_k=self.rp_top_k)
        
        # custom logger
        self.my_logger = my_logger

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = self.backbone_model(x)
        return embs

    def on_train_epoch_start(self):
        self.cross_entropy_loss.weight = self.weight_manager.get_weight().to(self.device)
        self.binary_cross_entropy_loss.pos_weight = self.weight_manager.get_pos_weight().to(self.device)

    def training_step(self, batch, batch_idx):
        # get anchors, positives, negatives
        x_a, y_a, binary_y_a = batch['anc']
        x_p, y_p, binary_y_p = batch['pos']
        x_n, y_n, binary_y_n = batch['neg']
        # embeddings
        z_a = self(x_a)
        z_p = self(x_p)
        z_n = self(x_n)
        # triplet loss
        triplet_loss = self.triplet_loss(z_a, z_p, z_n)
        # softmax on leaves
        logits = torch.cat([
            self.prediction_head_leaf(z_a),
            self.prediction_head_leaf(z_p),
            self.prediction_head_leaf(z_n)
        ], dim=0)
        target = torch.cat([y_a, y_p, y_n], dim=0)
        softmax_on_leaf = self.cross_entropy_loss(logits, target) * self.alpha
        # binary on all nodes except root
        binary_logits = torch.cat([
            self.prediction_head_all_nodes_except_root(z_a),
            self.prediction_head_all_nodes_except_root(z_p),
            self.prediction_head_all_nodes_except_root(z_n)
        ], dim=0)
        binary_target = torch.cat([binary_y_a, binary_y_p, binary_y_n], dim=0)
        binary_on_all_nodes_except_root = self.binary_cross_entropy_loss(
                                            binary_logits, binary_target) * self.beta
        # add
        loss = triplet_loss + softmax_on_leaf + binary_on_all_nodes_except_root
        # log
        self.log('train_loss/triplet', triplet_loss)
        self.log('train_loss/softmax_on_leaf', softmax_on_leaf)
        self.log('train_loss/binary_on_all_nodes_except_root', binary_on_all_nodes_except_root)
        self.log('train_loss/total', loss)
        return loss

    def on_validation_epoch_start(self):
        self.cross_entropy_loss.weight = None
        self.binary_cross_entropy_loss.pos_weight = None
        self.val_embeddings = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        x, y, binary_y = batch
        z = self(x)
        # softmax on leaves
        logits = self.prediction_head_leaf(z)
        val_softmax_on_leaf = self.cross_entropy_loss(logits, y)
        self.log('valid_loss/softmax_on_leaf', val_softmax_on_leaf)
        # binary on all nodes except root
        binary_logits = self.prediction_head_all_nodes_except_root(z)
        val_binary_on_all_nodes_except_root = self.binary_cross_entropy_loss(
                                              binary_logits, binary_y)
        self.log('valid_loss/binary_on_all_nodes_except_root', 
                 val_binary_on_all_nodes_except_root)
        # retrieval metrics will be evaluated on epoch end
        self.val_embeddings.append(z)
        self.val_labels.append(y)
        # update classification accuracy
        self.accuracy.update(logits, y)
        self.accuracy_top_k.update(logits, y)

    def on_validation_epoch_end(self):
        rp = self._compute_rp(self.val_embeddings, self.val_labels)
        adaptive_rp = self._compute_adaptive_rp(self.val_embeddings, self.val_labels)
        self.log('valid_metric/accuracy', self.accuracy.compute())
        self.log('valid_metric/accuracy_top_k', self.accuracy_top_k.compute())
        self.log('valid_metric/retrieval_precision', rp)
        self.log('valid_metric/adaptive_retrieval_precision', adaptive_rp)
        self.accuracy.reset()
        self.accuracy_top_k.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _compute_rp(
        self, 
        embs: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        embs = torch.cat(embs)
        labels = torch.cat(labels)
        sim_mtx = self.triplet_loss.distance.compute_mat(embs, embs) * \
                torch.tensor(1. if self.triplet_loss.distance.is_inverted else -1.)
        preds = torch.cat(
            [torch.cat((row[:i],row[i+1:])) for i, row in enumerate(sim_mtx)]
        )
        label_mtx = labels[:, None] == labels[None, :]
        target = torch.cat(
            [torch.cat((row[:i],row[i+1:])) for i, row in enumerate(label_mtx)]
        )
        N = embs.shape[0]
        indexes = torch.arange(N * (N - 1)) // (N - 1)
        return self.retrieval_precision(preds, target, indexes)

    def _compute_adaptive_rp(
        self, 
        embs: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        embs = torch.cat(embs)
        labels = torch.cat(labels)
        sim_mtx = self.triplet_loss.distance.compute_mat(embs, embs) * \
                torch.tensor(1. if self.triplet_loss.distance.is_inverted else -1.)
        label_mtx = labels[:, None] == labels[None, :]
        r_p = []
        for i in range(len(sim_mtx)):
            preds = torch.cat((sim_mtx[i,:i], sim_mtx[i,i+1:]))
            target = torch.cat((label_mtx[i,:i], label_mtx[i,i+1:]))
            total_relevant_num = int(target.sum())
            top_k = min(total_relevant_num, self.rp_top_k)
            if top_k > 0:
                r_p.append(retrieval_precision(preds, target, top_k=top_k))
        return torch.stack(r_p).mean()


if __name__ == '__main__':

    # example code

    import lightning as L
    L.seed_everything(2024)
    from label_metric.utils.log_utils import setup_logger
    logger = logging.getLogger(__name__)
    setup_logger(logger)

    weight_manager = WeightManager(logger, active = True)

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
        num_workers = 11
    )

    dm.setup('fit')

    # normalisatiion - todo: try this at the end
    #
    # from label_metric.datasets import BasicOrchideaSOL
    # collect_stat_train_set = BasicOrchideaSOL(
    #     dataset_dir = '/data/scratch/acw751/_OrchideaSOL2020_release',
    #     split = 'train',
    #     min_num_per_leaf = 10,
    #     duration = 1.0,
    #     train_ratio = 0.8,
    #     valid_ratio = 0.1,
    #     logger = logger
    # )
    # from torch.utils.data import DataLoader
    # collect_stat_train_loader = DataLoader(
    #     collect_stat_train_set,
    #     batch_size = 64,
    #     num_workers = 11,
    #     shuffle = False,
    #     drop_last = False
    # )
    # from label_metric.models import Audio2LogMelSpec
    # melspec = Audio2LogMelSpec(
    #     sr = 44100,
    #     n_fft = 2048,
    #     hop_length = 512
    # )
    # train_spec_max_val = float('-inf')
    # train_spec_min_val = float('inf')
    # for x, y1, y2 in collect_stat_train_loader:
    #     x = melspec(x)
    #     batch_max = x.max().item()
    #     batch_min = x.min().item()
    #     if batch_max > train_spec_max_val:
    #         train_spec_max_val = batch_max
    #     if batch_min < train_spec_min_val:
    #         train_spec_min_val = batch_min

    from label_metric.models import ConvModel, PredictionHead

    backbone_model = ConvModel(
        duration = 1.0,
        conv_out_channels = 128,
        embedding_size = 256,
        train_spec_max_val = None,
        train_spec_min_val = None,
        sr = 44100,
        n_fft = 2048,
        hop_length = 512
    )

    prediction_head_leaf = PredictionHead(
        embedding_size = 256,
        num_classes = dm.train_set.get_leaf_num()
    )

    prediction_head_all_nodes_except_root = PredictionHead(
        embedding_size = 256,
        num_classes = dm.train_set.get_node_num() - 1
    )

    from pytorch_metric_learning.distances import CosineSimilarity

    triplet_loss = TripletLoss(margin=0.3, distance=CosineSimilarity())

    lm = LabelMetricModule(
        backbone_model = backbone_model,
        prediction_head_leaf = prediction_head_leaf,
        prediction_head_all_nodes_except_root = prediction_head_all_nodes_except_root,
        triplet_loss = triplet_loss,
        alpha = 1.0,
        beta = 1.0,
        my_logger = logger,
        weight_manager = weight_manager,
        learning_rate = 0.001,
        classification_accuracy_top_k = 5,
        retrieval_precision_top_k = 5
    )

    trainer = L.Trainer(
        max_epochs = 500, 
        gradient_clip_val = 1.,
        enable_progress_bar = False,
        deterministic = True
    )
    trainer.fit(model = lm, datamodule = dm)

    # dm.setup('fit')
    # train_loader = dm.train_dataloader()
    # valid_loader = dm.val_dataloader()

    # lm.on_train_epoch_start()
    # for i, batch in enumerate(train_loader):
    #     lm.training_step(batch, batch_idx=i)

    # lm.on_validation_epoch_start()
    # for i, batch in enumerate(valid_loader):
    #     lm.validation_step(batch, batch_idx=i)
    # lm.on_validation_epoch_end()
    