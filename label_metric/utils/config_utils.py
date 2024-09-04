import yaml

import lightning as L
import torch.nn as nn
from pytorch_metric_learning.distances import CosineSimilarity
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from label_metric.data_modules import OrchideaSOLDataModule
from label_metric.models import ConvModel, PredictionHead
from label_metric.lightning_modules import LabelMetricModule
from label_metric.losses import TripletLoss

def get_trainer(config):
    logger = get_tblogger(config)
    return L.Trainer(
        callbacks=[EarlyStopping(
            monitor = 'valid_loss/total',
            patience = config['early_stopping_patience']
        )],
        max_epochs = config['max_epochs'],
        gradient_clip_val = config['gradient_clip_val'],
        enable_progress_bar = config['enable_progress_bar'],
        logger = logger,
        check_val_every_n_epoch = config['check_val_every_n_epoch'],
        deterministic = True
    )

def get_tblogger(config):
    return TensorBoardLogger(
        save_dir = config['save_dir'],
        name = config['name'],
        version = config['version']
    )

def get_model(config, dm, logger, weight_manager):
    if config['train_spec_max_val'] == 'None':
        config['train_spec_max_val'] = None
    if config['train_spec_min_val'] == 'None':
        config['train_spec_min_val'] = None
    backbone_model = get_backbone_model(config)
    prediction_heads = get_prediction_heads(config, dm)
    triplet_loss = TripletLoss(margin=config['margin'], distance=CosineSimilarity())
    return LabelMetricModule(
        backbone_model = backbone_model,
        prediction_heads = prediction_heads,
        triplet_loss = triplet_loss,
        use_triplet = config['use_triplet'],
        use_leaf = config['use_leaf'],
        use_binary = config['use_binary'],
        use_per_level = config['use_per_level'],
        my_logger = logger,
        weight_manager = weight_manager,
        learning_rate = config['learning_rate'],
        weight_decay = config['weight_decay'],
        lr_reduce_factor = config['lr_reduce_factor'],
        retrieval_precision_top_k = config['retrieval_precision_top_k'],
        lr_scheduler_patience = config['lr_scheduler_patience'],
        mask_value = config['mask_value']
    )

def get_backbone_model(config):
    return ConvModel(
        duration = config['duration'],
        conv_out_channels = config['conv_out_channels'],
        embedding_size = config['embedding_size'],
        train_spec_max_val = config['train_spec_max_val'],
        train_spec_min_val = config['train_spec_min_val'],
        sr = config['sr'],
        n_fft = config['n_fft'],
        hop_length = config['hop_length']
    )

def get_prediction_heads(config, dm):
    return nn.ModuleDict({
        'leaf': PredictionHead(
            embedding_size = config['embedding_size'],
            num_classes = len(dm.train_set.visible_leaves)
        ),
        'binary': PredictionHead(
            embedding_size = config['embedding_size'],
            num_classes = len(dm.train_set.level_order_flat_visible_nodes) - 1
        ),
        'per_level': nn.ModuleList([
            PredictionHead(embedding_size=config['embedding_size'], num_classes=len(nodes)) \
            for nodes in dm.train_set.level_order_visible_nodes if len(nodes) > 1
        ])
    })

def load_config(fpath):
    with open(fpath, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_dm(config, dataset_dir, logger, weight_manager):
    return OrchideaSOLDataModule(
        dataset_dir = dataset_dir,
        min_num_per_leaf = config['min_num_per_leaf'],
        duration = config['duration'],
        train_ratio = config['train_ratio'],
        valid_ratio = config['valid_ratio'],
        logger = logger,
        dataset_sr = config['dataset_sr'],
        dataset_channel_num = config['dataset_channel_num'],
        fold_num = config['fold_num'],
        fold_id = config['fold_id'],
        mask_value = config['mask_value'],
        random_seed = config['random_seed'],
        more_level = config['more_level'],
        weight_manager = weight_manager,
        batch_size = config['batch_size'],
        num_workers = config['num_workers']
    )
