import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

class Audio2MelSpec(nn.Module):
    def __init__(self, sr: int, n_fft: int, hop_length: int):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
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


class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels = 64, 
        conv_kernel_size = (3, 3),
        pool_kernel_size = (2, 2)
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel_size, padding='same')
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class ConvModel(nn.Module):
    def __init__(
        self, 
        duration: float, 
        conv_out_channels: int,
        embedding_size: int, 
        **kwargs
    ):
        super().__init__()
        self.duration = duration
        self.embedding_size = embedding_size
        self.melspec = Audio2MelSpec(**kwargs)
        conv_layers = [
            ConvBlock(in_channels = 1, 
                out_channels = conv_out_channels),
            ConvBlock(in_channels = conv_out_channels, 
                out_channels = conv_out_channels),
            ConvBlock(in_channels = conv_out_channels, 
                out_channels = conv_out_channels),
            ConvBlock(in_channels = conv_out_channels, 
                out_channels = conv_out_channels)
        ]
        self.conv = nn.Sequential(*conv_layers)
        shape = self._get_shape()
        self.pool = nn.MaxPool2d(kernel_size=(1, shape[-1]))
        self.project = nn.Linear(conv_out_channels * shape[-2], embedding_size)

    def _get_shape(self) -> torch.Tensor:
        _input_like = torch.randn([1, int(self.duration * self.melspec.sr)])
        x = self.melspec(_input_like).unsqueeze(1)
        return self.conv(x).shape

    def forward(self, x):
        x = self.melspec(x).unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.project(x)
        return x


class PredictionHead(nn.Module):
    def __init__(self, embedding_size: int, num_classes: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.linear = nn.Linear(embedding_size, num_classes)
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

    model = ConvModel(
        duration = 1.0,
        conv_out_channels = 128,
        embedding_size = 128,
        sr = 44100,
        n_fft = 2048,
        hop_length = 512,
    )
    
    y = model(x)

    print(f"output shape: {y.shape}")
