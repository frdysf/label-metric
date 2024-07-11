from typing import Type, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.distances import BaseDistance

class TripletLoss(nn.Module):
    def __init__(
        self, 
        margin: float, 
        distance: Type[BaseDistance]
    ):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self, 
        anchor_embs: torch.Tensor, 
        positive_embs: torch.Tensor,
        negative_embs: torch.Tensor
    ):
        d_ap, d_an = self._compute_pairwise_distance(
            anchor_embs, 
            positive_embs, 
            negative_embs
        )
        if self.distance.is_inverted:
            losses = F.relu(self.margin - d_ap + d_an)
        else:
            losses = F.relu(self.margin + d_ap - d_an)
        return losses.mean()

    def _compute_pairwise_distance(
        self, 
        anchor_embs: torch.Tensor, 
        positive_embs: torch.Tensor,
        negative_embs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # we don't use BaseDistance forward()
        # because it uses compute_mat(), too much computation
        assert anchor_embs.shape == positive_embs.shape == negative_embs.shape
        if self.distance.normalize_embeddings:
            anchor_embs = self.distance.normalize(anchor_embs)
            positive_embs = self.distance.normalize(positive_embs)
            negative_embs = self.distance.normalize(negative_embs)
        d_ap = self.distance.pairwise_distance(anchor_embs, positive_embs)
        d_an = self.distance.pairwise_distance(anchor_embs, negative_embs)
        return d_ap, d_an

if __name__ == '__main__':

    import lightning as L
    L.seed_everything(2024)

    from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
    from label_metric.distances import PoincareDistance

    batch_num = 32
    embedding_size = 256

    a = torch.randn([batch_num, embedding_size])
    p = torch.randn([batch_num, embedding_size])
    n = torch.randn([batch_num, embedding_size])

    loss_fn = TripletLoss(margin=0.1, distance=LpDistance())
    loss = loss_fn(a, p, n)
    print(loss)

    loss_fn = TripletLoss(margin=0.1, distance=CosineSimilarity())
    loss = loss_fn(a, p, n)
    print(loss)

    loss_fn = TripletLoss(margin=0.1, distance=PoincareDistance())
    loss = loss_fn(a, p, n)
    print(loss)

    a = torch.tensor([[0., 1.]])
    p = torch.tensor([[1., 0.]])
    n = torch.tensor([[0., 1.]])
    m = 0.5

    loss_fn = TripletLoss(margin=m, distance=LpDistance())
    loss = loss_fn(a, p, n)
    print(loss)

    loss_fn = TripletLoss(margin=m, distance=CosineSimilarity())
    loss = loss_fn(a, p, n)
    print(loss)

    loss_fn = TripletLoss(margin=m, distance=PoincareDistance())
    loss = loss_fn(a, p, n)
    print(loss)
