import logging

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance, BaseDistance

# TODO: add hyperbolic distance using BaseDistance

if __name__ == '__main__':

    # example code
    
    import lightning as L
    L.seed_everything(2024)
    
    from label_metric.utils.log_utils import setup_logger
    logger = logging.getLogger(__name__)
    setup_logger(logger)

    from label_metric.data_modules import OrchideaSOLDataModule

    data_module = OrchideaSOLDataModule(
        dataset_dir = '/data/scratch/acw751/_OrchideaSOL2020_release',
        min_num_per_leaf = 10,
        duration = 1.0,
        train_ratio = 0.8,
        valid_ratio = 0.1,
        logger = logger,
        more_level = 1,
        batch_size = 32, 
        num_workers = 2
    )

    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))

    from label_metric.models import PlaceHolderModel

    model = PlaceHolderModel(
        sr = 44100,
        n_fft = 2048,
        hop_length = 512,
        output_dim = 256
    )

    x_a = batch['anc'][0]
    x_p = batch['pos'][0]
    x_n = batch['neg'][0]

    y_a = model(x_a)
    y_p = model(x_p)
    y_n = model(x_n)

    print(f'anchor: {y_a.shape}, positive: {y_p.shape}, negative: {y_n.shape}')

    dist = LpDistance()

    D = dist.pairwise_distance(y_a, y_p)

    print(f'pairwise distance: {D.shape}')
