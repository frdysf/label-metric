from pytorch_metric_learning.distances import BaseDistance

from label_metric.hyperbolic import pmath

class PoincareDistance(BaseDistance):

    def __init__(self, c=1.0, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted
        self.c = c
    
    def compute_mat(self, query_emb, ref_emb):
        if ref_emb is None:
            ref_emb = query_emb
        return pmath.dist_matrix(query_emb, ref_emb, c=self.c)

    def pairwise_distance(self, query_emb, ref_emb):
        return pmath.dist(query_emb, ref_emb, c=self.c)

if __name__ == '__main__':

    # example code

    import logging
    from label_metric.utils.log_utils import setup_logger
    logger = logging.getLogger(__name__)
    setup_logger(logger)

    import lightning as L
    L.seed_everything(2024)

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
        more_level = 1,
        weight_manager = weight_manager,
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

    from pytorch_metric_learning.distances import LpDistance
    dist = LpDistance()
    D = dist.pairwise_distance(y_a, y_p)
    print(f'pairwise distance: {D.shape}')

    # this is a test, in practice PoincareDistance should only take poincare embeddings

    pdist = PoincareDistance()
    pD = pdist.pairwise_distance(y_a, y_p)
    print(f'hyperbolic distance: {pD.shape}')
