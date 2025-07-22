import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential
from torch.nn import LSTM

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential
from torch.nn import LSTM


class LstmEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.news_dim = 400

        self.lstm = nn.LSTM(self.news_dim, self.news_dim, num_layers=2, batch_first=True, bidirectional=True)

        # Add linear layer to project from 2*news_dim back to news_dim
        self.projection = nn.Linear(2 * self.news_dim, self.news_dim)

        self.atte = Sequential('x, mask', [
            (MultiHeadAttention(self.news_dim,
                                self.news_dim,
                                self.news_dim,
                                cfg.model.head_num,
                                cfg.model.head_dim), 'x,x,x,mask -> x'),
        ])

    def forward(self, clicked_origin_emb, clicked_mask=None):
        output, _ = self.lstm(clicked_origin_emb)
        output = self.projection(output)
        result = self.atte(output, clicked_mask)
        return result

