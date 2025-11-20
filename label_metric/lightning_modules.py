import logging
from typing import Tuple, Dict, Union, List

import torch
import torch.nn as nn
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
        weight_decay: float,
        lr_reduce_factor: float,
        retrieval_precision_top_k: int,
        lr_scheduler_patience: int,
        mask_value: int,
        compile: bool = False,
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
        self.weight_decay = weight_decay
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_reduce_factor = lr_reduce_factor
        
        # evaluation
        self.leaf_accuracy = Accuracy(
            task = 'multiclass', 
            num_classes = prediction_heads['leaf'].num_classes
        )
        self.leaf_f1 = F1Score(
            task = 'multiclass', 
            num_classes = prediction_heads['leaf'].num_classes
        )
        self.binary_accuracy = Accuracy(
            task = 'multilabel',
            num_labels = prediction_heads['binary'].num_classes
        )
        self.binary_f1 = F1Score(
            task = 'multilabel',
            num_labels = prediction_heads['binary'].num_classes
        )
        # note that in Orchidea SOL's tree structure, bottom level is equivalent to leaf level
        self.bottom_level_accuracy = Accuracy(
            task = 'multiclass',
            num_classes = prediction_heads['per_level'][-1].num_classes
        )
        self.bottom_level_f1 = F1Score(
            task = 'multiclass', 
            num_classes = prediction_heads['per_level'][-1].num_classes
        )
        self.rp_top_k = retrieval_precision_top_k
        self.retrieval_precision = RetrievalPrecision(top_k=self.rp_top_k)
        
        self.mask_value = mask_value
        
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
            self.log('train_loss/leaf', softmax_on_leaf)
        if self.loss_activations['binary']:
            self.log('train_loss/binary', binary_loss)
        if self.loss_activations['per_level']:
            self.log('train_loss/per_level', softmax_per_level)
        self.log('train_loss/total', loss)
        
        return loss
    
    def on_eval_epoch_start(self):
        self.clf_loss['leaf'].weight = None
        self.clf_loss['binary'].pos_weight = None
        for loss_fn in self.clf_loss['per_level']:
            loss_fn.weight = None
        self.eval_embeddings = []
        self.eval_labels = []
        self.eval_aff_idx = []
        self.eval_per_level_labels = []

    def on_validation_epoch_start(self):
        self.on_eval_epoch_start()
    
    def on_test_epoch_start(self):
        self.on_eval_epoch_start()

    def eval_step(self, batch, stage: str):

        x, y = batch
        z = self(x)
        
        # softmax on leaves
        logits = self.prediction_heads['leaf'](z)
        eval_softmax_on_leaf = self.clf_loss['leaf'](logits, y['leaf'])
        if self.loss_activations['leaf']:
            self.log(f'{stage}_loss/leaf', eval_softmax_on_leaf)
        
        # binary on all nodes except root
        binary_logits = self.prediction_heads['binary'](z)
        eval_binary_loss = self.clf_loss['binary'](binary_logits, y['binary'])
        if self.loss_activations['binary']:
            self.log(f'{stage}_loss/binary', eval_binary_loss)

        # softmax per level
        eval_softmax_per_level = torch.tensor(.0, device=self.device)
        level_num = len(self.prediction_heads['per_level'])
        for i in range(level_num):
            phead = self.prediction_heads['per_level'][i]
            eval_softmax_per_level += self.clf_loss['per_level'][i](phead(z), y['per_level'][:,i])
        bottom_level_logits = self.prediction_heads['per_level'][-1](z)
        if self.loss_activations['per_level']:
            self.log(f'{stage}_loss/per_level', eval_softmax_per_level)

        # total val loss
        total_eval_loss = eval_softmax_on_leaf * self.loss_activations['leaf'] + \
            eval_binary_loss * self.loss_activations['binary'] + \
            eval_softmax_per_level * self.loss_activations['per_level']
        self.log(f'{stage}_loss/total', total_eval_loss)
        
        # retrieval metrics will be computed on epoch end
        self.eval_embeddings.append(z)
        self.eval_labels.append(y['leaf'])
        self.eval_aff_idx.append(y['aff_idx'])
        self.eval_per_level_labels.append(y['per_level'])
        
        # update classification metrics
        self.leaf_accuracy.update(logits, y['leaf'])
        self.leaf_f1.update(logits, y['leaf'])
        self.binary_accuracy.update(binary_logits, y['binary'])
        self.binary_f1.update(binary_logits, y['binary'])
        self.bottom_level_accuracy.update(bottom_level_logits, y['per_level'][:,-1])
        self.bottom_level_f1.update(bottom_level_logits, y['per_level'][:,-1])

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, 'valid')

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, 'test')

    def on_eval_epoch_end(self, stage):
        self.eval_embeddings = torch.cat(self.eval_embeddings)
        self.eval_labels = torch.cat(self.eval_labels)
        self.eval_aff_idx = torch.cat(self.eval_aff_idx)
        self.eval_per_level_labels = torch.cat(self.eval_per_level_labels)
        # retrieval
        similarity_matrix = self.triplet_loss.distance.compute_mat(self.eval_embeddings, self.eval_embeddings) * \
            torch.tensor(1. if self.triplet_loss.distance.is_inverted else -1.)
        rp = self._compute_rp(similarity_matrix, self.eval_labels)
        adaptive_rp = self._compute_adaptive_rp(similarity_matrix, self.eval_labels)
        self.log(f'{stage}_metric/retrieval/precision@{self.rp_top_k}', rp)
        self.log(f'{stage}_metric/retrieval/adaptive_precision@{self.rp_top_k}', adaptive_rp)
        ndcg_max, ndcg_sum = self._compute_ndcg(similarity_matrix, self.eval_aff_idx)
        self.log(f'{stage}_metric/retrieval/ndcg_max', ndcg_max)
        self.log(f'{stage}_metric/retrieval/ndcg_sum', ndcg_sum)
        mnr = self._compute_mnr(similarity_matrix, self.eval_per_level_labels)
        self.log(f'{stage}_metric/retrieval/mnr', mnr)
        # classification
        if self.loss_activations['leaf']:
            self.log(f'{stage}_metric/leaf/accuracy', self.leaf_accuracy.compute())
            self.log(f'{stage}_metric/leaf/f1', self.leaf_f1.compute())
        if self.loss_activations['binary']:
            self.log(f'{stage}_metric/binary/accuracy', self.binary_accuracy.compute())
            self.log(f'{stage}_metric/binary/f1', self.binary_f1.compute())
        if self.loss_activations['per_level']:
            self.log(f'{stage}_metric/bottom_level/accuracy', self.bottom_level_accuracy.compute())
            self.log(f'{stage}_metric/bottom_level/f1', self.bottom_level_f1.compute())
        self.leaf_accuracy.reset()
        self.leaf_f1.reset()
        self.binary_accuracy.reset()
        self.binary_f1.reset()
        self.bottom_level_accuracy.reset()
        self.bottom_level_f1.reset()
        self.retrieval_precision.reset()
        torch.cuda.empty_cache()
        self.log('memory/allocated', torch.cuda.memory_allocated() / 1024 ** 2)
        self.log('memory/reserved', torch.cuda.memory_reserved() / 1024 ** 2)

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end('valid')

    def on_test_epoch_end(self):
        self.on_eval_epoch_end('test')

    def on_predict_epoch_start(self):
        self.predict_counter = {
            'leaf': torch.tensor(0., device=self.device),
            'per_level_leaf_head': torch.tensor(0., device=self.device),
            'per_level_lvp_head': torch.tensor(0., device=self.device),
            'total': torch.tensor(0., device=self.device)
        }
        self.predict_embeddings = []
        self.predict_aff_idx = []

    def predict_step(self, batch, batch_idx):
        
        x, y = batch
        z = self(x)

        self.predict_counter['total'] += x.shape[0]

        level_order_visible_nodes = self.trainer.datamodule.predict_set.level_order_visible_nodes
        level_order_visible_nodes = [nodes for nodes in level_order_visible_nodes if len(nodes) > 1]

        for i in range(x.shape[0]):
            per_level_label = y['per_level'][i]
            mask_index = (per_level_label == self.mask_value).nonzero(as_tuple=True)[0][0]
            level = mask_index - 1
            true_label = per_level_label[level]
            
            if self.loss_activations['per_level']:
                logits = self.prediction_heads['per_level'][level](z[i])
                pred = torch.argmax(logits)
                if pred == true_label:
                    self.predict_counter['per_level_lvp_head'] += 1
                logits = self.prediction_heads['per_level'][-1](z[i])
                pred_leaf_idx = torch.argmax(logits)
                node = self.trainer.datamodule.predict_set.visible_leaves[pred_leaf_idx]
                find_parent_times = len(per_level_label) - mask_index
                for _ in range(find_parent_times):
                    node = node.parent
                pred = level_order_visible_nodes[level].index(node)
                if pred == true_label:
                    self.predict_counter['per_level_leaf_head'] += 1

            if self.loss_activations['leaf']:
                logits = self.prediction_heads['leaf'](z[i])
                pred_leaf_idx = torch.argmax(logits)
                node = self.trainer.datamodule.predict_set.visible_leaves[pred_leaf_idx]
                find_parent_times = len(per_level_label) - mask_index
                for _ in range(find_parent_times):
                    node = node.parent
                pred = level_order_visible_nodes[level].index(node)
                if pred == true_label:
                    self.predict_counter['leaf'] += 1

        self.predict_embeddings.append(z)
        self.predict_aff_idx.append(y['aff_idx'])

    def on_predict_epoch_end(self):
        if self.loss_activations['leaf']:
            accu = self.predict_counter['leaf'] / self.predict_counter['total']
            self.my_logger.info(f'Accuracy of predicting LVP: {accu}')
        if self.loss_activations['per_level']:
            accu_lvp_head = self.predict_counter['per_level_lvp_head'] / self.predict_counter['total']
            accu_leaf_head = self.predict_counter['per_level_leaf_head'] / self.predict_counter['total']
            self.my_logger.info(f'Accuracy of predicting LVP: '
                                f'{accu_lvp_head} (LVP head), '
                                f'{accu_leaf_head} (leaf head)')
        self.predict_embeddings = torch.cat(self.predict_embeddings)
        self.predict_aff_idx = torch.cat(self.predict_aff_idx)
        similarity_matrix = self.triplet_loss.distance.compute_mat(self.predict_embeddings, self.predict_embeddings) * \
            torch.tensor(1. if self.triplet_loss.distance.is_inverted else -1.)
        ndcg_max, ndcg_sum = self._compute_ndcg(similarity_matrix, self.predict_aff_idx)
        self.my_logger.info(f'Predict set ndcg_max {ndcg_max}, ndcg_sum {ndcg_sum}')

    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.lr_scheduler_patience, 
                                                         factor=self.lr_reduce_factor)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid_loss/total',
            }
        }

    def _compute_mnr(self, sim_mtx: torch.Tensor, per_level_labels: torch.Tensor) -> torch.Tensor:
        num_samples, num_levels = per_level_labels.shape
        per_query_mnr = []
        for i in range(num_samples):
            sim_scores = sim_mtx[i]
            sim_scores = torch.cat([sim_scores[:i], sim_scores[i+1:]])
            _, ranked_indices = torch.sort(sim_scores, descending=True)
            per_level_mnr = []
            for j in range(num_levels):
                labels = torch.cat([per_level_labels[:i, j], per_level_labels[i+1:, j]])
                query_label = per_level_labels[i, j]
                same_label_indices = (labels == query_label).nonzero(as_tuple=True)[0]
                same_label_ranks = (ranked_indices.unsqueeze(1) == same_label_indices).nonzero(as_tuple=True)[0]
                per_level_mnr.append(torch.mean(same_label_ranks / (num_samples - 1)))
            # nanmean for queries that do not have correct answers (a leaf only has one sample)
            per_query_mnr.append(torch.nanmean(torch.tensor(per_level_mnr)))
        return torch.mean(torch.tensor(per_query_mnr))

    def _compute_rp(self, sim_mtx: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds = torch.cat(
            [torch.cat((row[:i],row[i+1:])) for i, row in enumerate(sim_mtx)]
        )
        label_mtx = labels[:, None] == labels[None, :]
        target = torch.cat(
            [torch.cat((row[:i],row[i+1:])) for i, row in enumerate(label_mtx)]
        )
        N = len(sim_mtx)
        indexes = torch.arange(N * (N - 1)) // (N - 1)
        return self.retrieval_precision(preds, target, indexes)

    def _compute_adaptive_rp(self, sim_mtx: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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

    def _compute_ndcg(self, sim_mtx: torch.Tensor, aff_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = len(sim_mtx)
        mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
        sim_mtx = sim_mtx[mask].reshape(N, N-1)
        true_rel_sum = self.node_aff_sum[aff_idx][:,aff_idx]
        true_rel_max = self.node_aff_max[aff_idx][:,aff_idx]
        true_rel_sum = true_rel_sum[mask].reshape(N, N-1)
        true_rel_max = true_rel_max[mask].reshape(N, N-1)
        # retrieval_normalized_dcg does not support deterministic=True
        torch.use_deterministic_algorithms(False)
        ndcg_max = retrieval_normalized_dcg(sim_mtx, true_rel_max)
        ndcg_sum = retrieval_normalized_dcg(sim_mtx, true_rel_sum)
        torch.use_deterministic_algorithms(True)
        return ndcg_max, ndcg_sum
