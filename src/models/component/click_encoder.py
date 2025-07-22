import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential


class ClickEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.news_dim = 400
        self.atte = Sequential('a', [
            (lambda a: a.view(-1, 1, self.news_dim), 'a -> x'),
            AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        ])

    def forward(self, click_total_emb):
        batch_size, num_news = click_total_emb.shape[0], click_total_emb.shape[1]
        result = self.atte(click_total_emb)
        return result.view(batch_size, num_news, self.news_dim)

