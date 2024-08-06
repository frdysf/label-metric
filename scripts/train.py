from lightning.pytorch.cli import LightningCLI

from label_metric.data_modules import OrchideaSOLDataModule
from label_metric.lightning_modules import LabelMetricModule

if __name__ == '__main__':
    cli = LightningCLI(LabelMetricModule, OrchideaSOLDataModule)