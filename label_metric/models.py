import logging

import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

class Audio2MelSpec(nn.Module):

    def __init__(self, sr: int, n_fft: int, hop_length: int):
        super().__init__()
        self.melspec = MelSpectrogram(
            sample_rate = sr,
            n_fft = n_fft,
            hop_length = hop_length
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.melspec(x)
        return x


class PlaceHolderModel(nn.Module):

    def __init__(self, output_dim: int, **kwargs):

        super().__init__()

        self.output_dim = output_dim
        self.melspec = Audio2MelSpec(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.melspec(x)
        x = torch.reshape(x, (x.shape[0], -1))
        self.linear = nn.Linear(x.shape[1], self.output_dim)
        x = self.linear(x)

        return x


class PredictionHead(nn.module):

    def __init__(self, embedding_size: int, num_classes: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.linear = nn.Linear(emb_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    

if __name__ == '__main__':

    # example code
    
    import lightning as L
    L.seed_everything(2024)
    
    from label_metric.utils.log_utils import setup_logger
    logger = logging.getLogger(__name__)
    setup_logger(logger)

    from label_metric.data_modules import OrchideaSOLDataModule
    from label_metric.samplers import WeightManager

    weight_manager = WeightManager(logger)

    data_module = OrchideaSOLDataModule(
        dataset_dir = '/data/scratch/acw751/_OrchideaSOL2020_release',
        min_num_per_leaf = 10,
        duration = 1.0,
        train_ratio = 0.8,
        valid_ratio = 0.1,
        logger = logger,
        more_level = 2,
        weight_manager = weight_manager,
        batch_size = 32, 
        num_workers = 2
    )

    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))

    model = PlaceHolderModel(
        sr = 44100,
        n_fft = 2048,
        hop_length = 512,
        output_dim = 256
    )

    x = batch['anc'][0]
    y = model(x)

    print(f"input shape: {x.shape}, output shape: {y.shape}")
