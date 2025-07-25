import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential


class CandidateEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_entity = cfg.model.use_entity

        self.entity_dim = 100
        self.news_dim = cfg.model.head_dim * cfg.model.head_num
        self.output_dim = cfg.model.head_dim * cfg.model.head_num

        self.atte = Sequential('a,b,c', [
            (nn.Linear(self.news_dim, self.output_dim), 'a -> x'),
            nn.LeakyReLU(0.2)])


    def forward(self, candidate_emb, origin_emb=None, neighbor_emb=None):
        batch_size, num_news = candidate_emb.shape[0], candidate_emb.shape[1]
        result = self.atte(candidate_emb, origin_emb, neighbor_emb).view(batch_size, num_news, self.output_dim)
        return result
