import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torchmetrics import Accuracy, F1Score
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
        gamma: float,
        my_logger: logging.Logger,
        weight_manager: WeightManager,
        learning_rate: float,
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
        assert alpha >= 0 and beta >= 0 and gamma >= 0
        assert alpha + beta + gamma > 0
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.gamma = torch.tensor(gamma)
        
        # class weight manager
        self.weight_manager = weight_manager
        
        # optimization
        self.learning_rate = learning_rate
        
        # evaluation
        self.leaf_head_accuracy = Accuracy(
            task = 'multiclass', 
            num_classes = prediction_head_leaf.num_classes
        )
        self.leaf_head_f1 = F1Score(
            task = 'multiclass', 
            num_classes = prediction_head_leaf.num_classes
        )
        self.ANER_head_accuracy = Accuracy(
            task = 'multilabel',
            num_labels = prediction_head_all_nodes_except_root.num_classes
        )
        self.ANER_head_f1 = F1Score(
            task = 'multilabel',
            num_labels = prediction_head_all_nodes_except_root.num_classes
        )
        self.rp_top_k = retrieval_precision_top_k
        self.retrieval_precision = RetrievalPrecision(top_k=self.rp_top_k)
        
        # custom logger
        self.my_logger = my_logger

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = self.backbone_model(x)
        return embs

    # def on_fit_start(self):
    #     dm = self.trainer.datamodule
    #     self.my_logger.info(f'hi, this is fit start, {dm.train_set.get_leaf_num()}')

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
        triplet_loss = self.triplet_loss(z_a, z_p, z_n) * self.alpha
        
        # softmax on leaves
        logits = torch.cat([
            self.prediction_head_leaf(z_a),
            self.prediction_head_leaf(z_p),
            self.prediction_head_leaf(z_n)
        ], dim=0)
        target = torch.cat([y_a, y_p, y_n], dim=0)
        softmax_on_leaf = self.cross_entropy_loss(logits, target) * self.beta
        
        # binary on all nodes except root
        binary_logits = torch.cat([
            self.prediction_head_all_nodes_except_root(z_a),
            self.prediction_head_all_nodes_except_root(z_p),
            self.prediction_head_all_nodes_except_root(z_n)
        ], dim=0)
        binary_target = torch.cat([binary_y_a, binary_y_p, binary_y_n], dim=0)
        binary_on_all_nodes_except_root = self.binary_cross_entropy_loss(
            binary_logits, binary_target) * self.gamma
        
        # add
        loss = triplet_loss + softmax_on_leaf + binary_on_all_nodes_except_root
        
        # log
        if self.alpha > 0:
            self.log('train_loss/triplet', triplet_loss)
        if self.beta > 0:
            self.log('train_loss/softmax_on_leaf', softmax_on_leaf)
        if self.gamma > 0:
            self.log('train_loss/binary_on_all_nodes_except_root', 
                binary_on_all_nodes_except_root)
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
        if self.beta > 0:
            self.log('valid_loss/softmax_on_leaf', 
                val_softmax_on_leaf)
        
        # binary on all nodes except root
        binary_logits = self.prediction_head_all_nodes_except_root(z)
        val_binary_on_all_nodes_except_root = self.binary_cross_entropy_loss(
            binary_logits, binary_y)
        if self.gamma:
            self.log('valid_loss/binary_on_all_nodes_except_root', 
                val_binary_on_all_nodes_except_root)
        
        # retrieval metrics will be computed on epoch end
        self.val_embeddings.append(z)
        self.val_labels.append(y)
        
        # update classification metrics
        self.leaf_head_accuracy.update(logits, y)
        self.ANER_head_accuracy.update(binary_logits, binary_y)
        self.leaf_head_f1.update(logits, y)
        self.ANER_head_f1.update(binary_logits, binary_y)

    def on_validation_epoch_end(self):
        # retrieval
        rp = self._compute_rp(self.val_embeddings, self.val_labels)
        adaptive_rp = self._compute_adaptive_rp(self.val_embeddings, self.val_labels)
        self.log(f'valid_metric/retrieval/precision@{self.rp_top_k}', rp)
        self.log(f'valid_metric/retrieval/adaptive_precision@{self.rp_top_k}', adaptive_rp)
        # classification
        if self.beta > 0:
            self.log('valid_metric/leaf_head/accuracy', self.leaf_head_accuracy.compute())
            self.log('valid_metric/leaf_head/f1', self.leaf_head_f1.compute())
        if self.gamma > 0:
            self.log('valid_metric/all_nodes_except_root_binary_head/accuracy', 
                self.ANER_head_accuracy.compute())
            self.log('valid_metric/all_nodes_except_root_binary_head/f1', 
                self.ANER_head_f1.compute())
        self.leaf_head_accuracy.reset()
        self.leaf_head_f1.reset()
        self.ANER_head_accuracy.reset()
        self.ANER_head_f1.reset()
        torch.cuda.empty_cache()
        self.log('memory/allocated', torch.cuda.memory_allocated() / 1024 ** 2)
        self.log('memory/reserved', torch.cuda.memory_reserved() / 1024 ** 2)

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
        gamma = 1.0,
        my_logger = logger,
        weight_manager = weight_manager,
        learning_rate = 0.001,
        retrieval_precision_top_k = 5
    )

    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(
        save_dir = 'lightning_logs', 
        name = 'exp2', 
        version = 'S_B_T_m=.3_memory_empty_cache_check_val_every_10_epoch_no_clf_metrics'
    )

    # from lightning.pytorch.profilers import PyTorchProfiler
    # profiler = PyTorchProfiler(
    #     on_trace_ready = torch.profiler.tensorboard_trace_handler('lightning_logs/profile0')
    # )

    trainer = L.Trainer(
        max_epochs = 3000, 
        gradient_clip_val = 1.,
        enable_progress_bar = False,
        deterministic = True,
        logger = logger,
        check_val_every_n_epoch = 10
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
    