import torch
import torch.nn as nn
import torch.nn.functional as F


class Interest(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        self.dense = nn.Linear(self.news_dim, 1)

    def forward(self, s_t, s_g):
        gamma = torch.sigmoid(self.dense(s_t))
        u = gamma * s_t + (1 - gamma) * s_g
        return u


