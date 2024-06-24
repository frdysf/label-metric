import logging

import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

class BaseBackbone(nn.Module):

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop_length: int
    ) -> None:

        super().__init__()

        self.melspec = MelSpectrogram(
            sample_rate = sr,
            n_fft = n_fft,
            hop_length = hop_length
        )

    def forward(self, x):

        x = self.melspec(x)

        return x


class PlaceHolderModel(BaseBackbone):

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop_length: int,
        output_dim: int
    ) -> None:

        super().__init__(
            sr = sr,
            n_fft = n_fft,
            hop_length = hop_length
        )

        n_mels = 128
        n_times = torch.ceil(torch.tensor(44100 / hop_length)).int()

        self.linear = nn.Linear(n_mels * n_times, output_dim)

    def forward(self, x):

        x = self.melspec(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.linear(x)

        return x
    

if __name__ == '__main__':

    # example code
    
    import lightning as L
    from torch.utils.data import DataLoader

    from label_metric.datasets import TripletOrchideaSOL, BasicOrchideaSOL
    from label_metric.samplers import SampleTripletsFromTree
    from label_metric.utils.log_utils import setup_logger

    logger = logging.getLogger(__name__)
    setup_logger(logger)

    L.seed_everything(2024)

    train_set = TripletOrchideaSOL(
        dataset_dir = '/data/scratch/acw751/_OrchideaSOL2020_release',
        split = 'train',
        min_num_per_leaf = 10,
        duration = 1.0,
        train_ratio = 0.8,
        valid_ratio = 0.1,
        logger = logger
    )

    sampler = SampleTripletsFromTree(
        data = train_set, 
        more_level = 0,
        logger = logger
    )

    train_loader = DataLoader(
        train_set,
        batch_size = 32,
        sampler = sampler,
        num_workers = 0,
        drop_last = True
    )
    
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
