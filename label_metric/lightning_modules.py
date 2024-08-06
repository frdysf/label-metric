import logging
from typing import Optional, Tuple, Dict, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torchmetrics import Accuracy, F1Score
from torchmetrics.retrieval import RetrievalPrecision
from torchmetrics.functional.retrieval import retrieval_precision, retrieval_normalized_dcg

from label_metric.samplers import WeightManager
from label_metric.losses import TripletLoss

class LabelMetricModule(L.LightningModule):

    def __init__(
        self,
        backbone_model: nn.Module,
        prediction_heads: Dict[str, Union[nn.Module, List[nn.Module]]],
        triplet_loss: TripletLoss,
        use_triplet: bool,
        use_leaf: bool,
        use_binary: bool,
        use_per_level: bool,
        my_logger: logging.Logger,
        weight_manager: WeightManager,
        learning_rate: float,
        retrieval_precision_top_k: int
    ):
        super().__init__()
        
        # models
        self.backbone_model = backbone_model
        self.prediction_heads = prediction_heads
        
        # losses
        self.triplet_loss = triplet_loss
        self.clf_loss = {
            'leaf': nn.CrossEntropyLoss(),
            'binary': nn.BCEWithLogitsLoss(),
            'per_level': [nn.CrossEntropyLoss() for _ in \
                range(len(prediction_heads['per_level']))]
        }
        
        # loss activations
        self.loss_activations = {
            'triplet':      torch.tensor(float(use_triplet), device=self.device),
            'leaf':         torch.tensor(float(use_leaf), device=self.device),
            'binary':       torch.tensor(float(use_binary), device=self.device),
            'per_level':    torch.tensor(float(use_per_level), device=self.device)
        }
        
        # class weight manager
        self.weight_manager = weight_manager
        
        # optimization
        self.learning_rate = learning_rate
        
        # evaluation
        self.leaf_head_accuracy = Accuracy(
            task = 'multiclass', 
            num_classes = prediction_heads['leaf'].num_classes
        )
        self.leaf_head_f1 = F1Score(
            task = 'multiclass', 
            num_classes = prediction_heads['leaf'].num_classes
        )
        self.ANER_head_accuracy = Accuracy(
            task = 'multilabel',
            num_labels = prediction_heads['binary'].num_classes
        )
        self.ANER_head_f1 = F1Score(
            task = 'multilabel',
            num_labels = prediction_heads['binary'].num_classes
        )
        self.rp_top_k = retrieval_precision_top_k
        self.retrieval_precision = RetrievalPrecision(top_k=self.rp_top_k)
        
        # custom logger
        self.my_logger = my_logger

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = self.backbone_model(x)
        return embs

    def on_fit_start(self):
        self.node_aff_sum = self.trainer.datamodule.valid_set.aff_mtx['sum'].to(self.device)
        self.node_aff_max = self.trainer.datamodule.valid_set.aff_mtx['max'].to(self.device)

    def on_train_epoch_start(self):
        w = self.weight_manager.get_weight()
        self.clf_loss['leaf'].weight = w['leaf'].to(self.device)
        self.clf_loss['binary'].pos_weight = w['binary'].to(self.device)
        for idx, loss in enumerate(self.clf_loss['per_level']):
            loss.weight = w['per_level'][idx].to(self.device)

    def training_step(self, batch, batch_idx):
        
        # get anchors, positives, negatives
        x_a, y_a = batch['anc']
        x_p, y_p = batch['pos']
        x_n, y_n = batch['neg']
        
        # embeddings
        z_a = self(x_a)
        z_p = self(x_p)
        z_n = self(x_n)
        
        # triplet loss
        triplet_loss = self.triplet_loss(z_a, z_p, z_n) * self.loss_activations['triplet']
        
        # softmax on leaves
        logits = torch.cat([
            self.prediction_heads['leaf'](z_a),
            self.prediction_heads['leaf'](z_p),
            self.prediction_heads['leaf'](z_n)
        ], dim=0)
        target = torch.cat([y_a['leaf'], y_p['leaf'], y_n['leaf']], dim=0)
        softmax_on_leaf = self.clf_loss['leaf'](logits, target) * self.loss_activations['leaf']
        
        # binary on all nodes except root
        binary_logits = torch.cat([
            self.prediction_heads['binary'](z_a),
            self.prediction_heads['binary'](z_p),
            self.prediction_heads['binary'](z_n)
        ], dim=0)
        binary_target = torch.cat([y_a['binary'], y_p['binary'], y_n['binary']], dim=0)
        binary_loss = self.clf_loss['binary'](binary_logits, binary_target) * self.loss_activations['binary']
        
        # softmax per level
        softmax_per_level = torch.tensor(.0, device=self.device)
        level_num = len(self.prediction_heads['per_level'])
        for i in range(level_num):
            phead = self.prediction_heads['per_level'][i]
            logits = torch.cat([phead(z_a), phead(z_p), phead(z_n)], dim=0)
            target = torch.cat([
                y_a['per_level'][:,i], 
                y_p['per_level'][:,i], 
                y_n['per_level'][:,i]
            ], dim=0)
            softmax_per_level += self.clf_loss['per_level'][i](logits, target)
        softmax_per_level *= self.loss_activations['per_level']

        # add
        loss = triplet_loss + softmax_on_leaf + binary_loss + softmax_per_level
        
        # log
        if self.loss_activations['triplet']:
            self.log('train_loss/triplet', triplet_loss)
        if self.loss_activations['leaf']:
            self.log('train_loss/softmax_on_leaf', softmax_on_leaf)
        if self.loss_activations['binary']:
            self.log('train_loss/binary', binary_loss)
        if self.loss_activations['per_level']:
            self.log('train_loss/per_level', softmax_per_level)
        self.log('train_loss/total', loss)
        
        return loss

    def on_validation_epoch_start(self):
        self.clf_loss['leaf'].weight = None
        self.clf_loss['binary'].pos_weight = None
        for loss_fn in self.clf_loss['per_level']:
            loss_fn.weight = None
        self.val_embeddings = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        z = self(x)
        
        # softmax on leaves
        logits = self.prediction_heads['leaf'](z)
        val_softmax_on_leaf = self.clf_loss['leaf'](logits, y['leaf'])
        if self.loss_activations['leaf']:
            self.log('valid_loss/softmax_on_leaf', val_softmax_on_leaf)
        
        # binary on all nodes except root
        binary_logits = self.prediction_heads['binary'](z)
        val_binary_loss = self.clf_loss['binary'](binary_logits, y['binary'])
        if self.loss_activations['binary']:
            self.log('valid_loss/binary', val_binary_loss)

        # softmax per level
        val_softmax_per_level = torch.tensor(.0, device=self.device)
        level_num = len(self.prediction_heads['per_level'])
        for i in range(level_num):
            phead = self.prediction_heads['per_level'][i]
            val_softmax_per_level += self.clf_loss['per_level'][i](phead(z), y['per_level'][:,i])
        if self.loss_activations['per_level']:
            self.log('valid_loss/per_level', val_softmax_per_level)
        
        # retrieval metrics will be computed on epoch end
        self.val_embeddings.append(z)
        self.val_labels.append(y['leaf'])
        
        # update classification metrics
        self.leaf_head_accuracy.update(logits, y['leaf'])
        self.ANER_head_accuracy.update(binary_logits, y['binary'])
        self.leaf_head_f1.update(logits, y['leaf'])
        self.ANER_head_f1.update(binary_logits, y['binary'])

    def on_validation_epoch_end(self):
        # retrieval
        rp = self._compute_rp(self.val_embeddings, self.val_labels)
        adaptive_rp = self._compute_adaptive_rp(self.val_embeddings, self.val_labels)
        self.log(f'valid_metric/retrieval/precision@{self.rp_top_k}', rp)
        self.log(f'valid_metric/retrieval/adaptive_precision@{self.rp_top_k}', adaptive_rp)
        ndcg_max, ndcg_sum = self._compute_ndcg(self.val_embeddings, self.val_labels)
        self.log(f'valid_metric/retrieval/ndcg_max', ndcg_max)
        self.log(f'valid_metric/retrieval/ndcg_sum', ndcg_sum)
        # classification
        if self.loss_activations['leaf']:
            self.log('valid_metric/leaf_head/accuracy', self.leaf_head_accuracy.compute())
            self.log('valid_metric/leaf_head/f1', self.leaf_head_f1.compute())
        if self.loss_activations['binary']:
            self.log('valid_metric/all_nodes_except_root_binary_head/accuracy', 
                self.ANER_head_accuracy.compute())
            self.log('valid_metric/all_nodes_except_root_binary_head/f1', 
                self.ANER_head_f1.compute())
        self.leaf_head_accuracy.reset()
        self.leaf_head_f1.reset()
        self.ANER_head_accuracy.reset()
        self.ANER_head_f1.reset()
        self.retrieval_precision.reset()
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

    def _compute_ndcg(
            self, 
            embs: torch.Tensor, 
            labels: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        embs = torch.cat(embs)
        labels = torch.cat(labels)
        # this is similarity matrix, but it can have negative values
        sim_mtx = self.triplet_loss.distance.compute_mat(embs, embs) * \
            torch.tensor(1. if self.triplet_loss.distance.is_inverted else -1.)
        N = len(sim_mtx)
        mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
        sim_mtx = sim_mtx[mask].reshape(N, N-1)
        true_rel_sum = self.node_aff_sum[labels][:,labels]
        true_rel_max = self.node_aff_max[labels][:,labels]
        true_rel_sum = true_rel_sum[mask].reshape(N, N-1)
        true_rel_max = true_rel_max[mask].reshape(N, N-1)
        ndcg_max = retrieval_normalized_dcg(sim_mtx, true_rel_max)
        ndcg_sum = retrieval_normalized_dcg(sim_mtx, true_rel_sum)
        return ndcg_max, ndcg_sum


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
        dataset_sr = 44100,
        dataset_channel_num = 1,
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

    prediction_heads = nn.ModuleDict({
        'leaf': PredictionHead(
            embedding_size = 256,
            num_classes = dm.train_set.get_leaf_num()
        ),
        'binary': PredictionHead(
            embedding_size = 256,
            num_classes = dm.train_set.get_node_num() - 1
        ),
        'per_level': nn.ModuleList([
            PredictionHead(embedding_size=256, num_classes=size) \
            for size in dm.train_set.get_level_sizes()
        ])
    })

    from pytorch_metric_learning.distances import CosineSimilarity

    triplet_loss = TripletLoss(margin=0.3, distance=CosineSimilarity())

    lm = LabelMetricModule(
        backbone_model = backbone_model,
        prediction_heads = prediction_heads,
        triplet_loss = triplet_loss,
        use_triplet = True,
        use_leaf = True,
        use_binary = True,
        use_per_level = False,
        my_logger = logger,
        weight_manager = weight_manager,
        learning_rate = 0.001,
        retrieval_precision_top_k = 5
    )

    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(
        save_dir = 'lightning_logs', 
        name = 'exp4', 
        version = 'Leaf + Binary + Triplet(m=0.3)'
    )

    # from lightning.pytorch.profilers import PyTorchProfiler
    # profiler = PyTorchProfiler(
    #     on_trace_ready = torch.profiler.tensorboard_trace_handler('lightning_logs/profile0')
    # )

    trainer = L.Trainer(
        max_epochs = 3000, 
        gradient_clip_val = 1.,
        enable_progress_bar = False,
        logger = logger,
        check_val_every_n_epoch = 5
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
