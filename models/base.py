import torch
from torch import nn, Tensor
import typing as t


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def full_item_scores(self, user_ids: Tensor) -> Tensor:
        raise NotImplementedError("Full item scores method not implemented.")

    def l2_regularization(self, user_ids: Tensor, pos_item_ids: Tensor, neg_item_ids: t.Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError("L2 regularization method not implemented.")
