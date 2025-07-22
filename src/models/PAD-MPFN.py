import torch
import torch.nn as nn
from models.component.candidate_encoder import *
from models.component.click_encoder import ClickEncoder
from models.component.interest_encoder import Interest
from models.component.nce_loss import NCELoss
from models.component.news_encoder import *
from models.component.popular_encoder import PopularityEncoder
from models.component.sequence_encoder import LstmEncoder
from models.component.user_encoder import *
from models.component.neighbor_aggregator import UserNeighborAggregator
from torch_geometric.nn import Sequential, GatedGraphConv



class PAD-MPFN(nn.Module):
    def __init__(self, cfg, glove_emb=None, entity_emb=None):
        super().__init__()

        self.cfg = cfg

        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        self.entity_dim = cfg.model.entity_emb_dim

        # -------------------------- Model --------------------------
        # News Encoder
        self.local_news_encoder = NewsEncoder(cfg, glove_emb)
        # GCN
        self.global_news_encoder = Sequential('x, index', [
            (GatedGraphConv(self.news_dim, num_layers=3, aggr='add'), 'x, index -> x'),
        ])
        
        # LSTM
        self.sequence_encoder = LstmEncoder(cfg)


        # Click Encoder
        self.click_encoder = ClickEncoder(cfg)

        self.interest = Interest(cfg)

        self.user_encoder = UserEncoder(cfg)

        self.popularity_encoder = PopularityEncoder(cfg)

        # User Neighbor Aggregator (NUPIE)
        if cfg.model.use_neighbor_aggregation:
            self.neighbor_aggregator = UserNeighborAggregator(cfg)
        else:
            self.neighbor_aggregator = None

        self.candidate_encoder = CandidateEncoder(cfg)
        self.weight_layer = nn.Linear(3 * self.news_dim, 3)
        # click prediction
        self.click_predictor = DotProduct()
        self.loss_fn = NCELoss()

    def forward(self, subgraph, mapping_idx, candidate_news, pop_value, label=None, user_ids=None):
        # -------------------------------------- clicked ----------------------------------
        mask = mapping_idx != -1
        mapping_idx[mapping_idx == -1] = 0
        batch_size, num_clicked, token_dim = mapping_idx.shape[0], mapping_idx.shape[1], candidate_news.shape[-1]
        # News Encoder + GCN
        x_flatten = subgraph.x.view(1, -1, token_dim)
        x_encoded = self.local_news_encoder(x_flatten).view(-1, self.news_dim)
        graph_emb = self.global_news_encoder(x_encoded, subgraph.edge_index)
        clicked_origin_emb = x_encoded[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked,
                                                                                                self.news_dim)

        clicked_sequence_emb = self.click_encoder(clicked_origin_emb)

        clicked_emb = self.sequence_encoder(clicked_sequence_emb, mask)
        clicked_graph_emb = graph_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked,
                                                                                               self.news_dim)
        user_s_emb = self.user_encoder(clicked_emb, mask)

        user_g_emb = self.user_encoder(clicked_graph_emb, mask)
        user_emb = self.interest(user_s_emb, user_g_emb)

        # -------------------------------------- Neighbor Aggregation (NUPIE) ----------------------------------
        if self.neighbor_aggregator is not None and user_ids is not None:
            nei_user_emb = self.neighbor_aggregator(user_emb, user_ids)
        else:
            nei_user_emb = user_emb

        news_popularity = pop_value
        popular_emb = self.popularity_encoder(clicked_sequence_emb, news_popularity)

        cand_title_emb = self.local_news_encoder(candidate_news)
        cand_final_emb = self.candidate_encoder(cand_title_emb, None, None)

        y_u = self.click_predictor(cand_final_emb, user_emb)
        y_n = self.click_predictor(cand_final_emb, nei_user_emb)
        y_p = self.click_predictor(cand_final_emb, popular_emb)

        combined = torch.cat([user_emb, nei_user_emb, popular_emb], dim=-1)  # [32, 1200]
        weights = torch.softmax(self.weight_layer(combined), dim=-1)  # [32, 3]
        weight_u = weights[:, 0].unsqueeze(-1)  # [32, 1]
        weight_n = weights[:, 1].unsqueeze(-1)  # [32, 1]
        weight_p = weights[:, 2].unsqueeze(-1)

        score = weight_u * y_u + weight_n * y_n + weight_p * y_p

        if self.neighbor_aggregator is not None:
            neighbor_reg = 0.0
            for param in self.neighbor_aggregator.parameters():
                neighbor_reg += torch.norm(param, p=2)
            neighbor_reg = neighbor_reg * 1e-6
            loss = self.loss_fn(score, label) + neighbor_reg
        else:
            loss = self.loss_fn(score, label)

        return loss, score

    def validation_process(self, subgraph, mappings, candidate_emb, news_p, user_ids=None):
        device = next(self.parameters()).device
        news_p = news_p.to(device)
        batch_size, num_news, news_dim = 1, len(mappings), candidate_emb.shape[-1]

        title_graph_emb = self.global_news_encoder(subgraph.x, subgraph.edge_index)
        clicked_graph_emb = title_graph_emb[mappings, :].view(batch_size, num_news, news_dim)

        clicked_origin_emb = subgraph.x[mappings, :].view(batch_size, num_news, news_dim)

        clicked_sequence_emb = self.click_encoder(clicked_origin_emb)

        clicked_emb = self.sequence_encoder(clicked_sequence_emb)
        user_s_emb = self.user_encoder(clicked_emb)
        user_g_emb = self.user_encoder(clicked_graph_emb)
        user_emb = self.interest(user_s_emb, user_g_emb)

        # -------------------------------------- Neighbor Aggregation (NUPIE) ----------------------------------
        if self.neighbor_aggregator is not None and user_ids is not None:
            nei_user_emb = self.neighbor_aggregator(user_emb, user_ids)
        else:
            nei_user_emb = user_emb

        news_popularity = news_p
        popular_emb = self.popularity_encoder(clicked_sequence_emb, news_popularity)
        # ----------------------------------------- Candidate------------------------------------
        cand_final_emb = self.candidate_encoder(candidate_emb.unsqueeze(0), None, None)

        scores_u = self.click_predictor(cand_final_emb, user_emb)
        scores_n = self.click_predictor(cand_final_emb, nei_user_emb)
        scores_p = self.click_predictor(cand_final_emb, popular_emb)

        combined = torch.cat([user_emb, nei_user_emb, popular_emb], dim=-1)
        weights = torch.softmax(self.weight_layer(combined), dim=-1)  # [1, 3]
        weight_u = weights[:, 0].unsqueeze(-1)  # [1, 1]
        weight_n = weights[:, 1].unsqueeze(-1)  # [1, 1]
        weight_p = weights[:, 2].unsqueeze(-1)  # [1, 1]

        weighted_scores = weight_u * scores_u + weight_n * scores_n + weight_p * scores_p  # [1, num_candidates]

        return weighted_scores.view(-1).cpu().tolist()

    def save_user_cache(self, save_path):
        """保存用户缓存"""
        if self.neighbor_aggregator is not None:
            self.neighbor_aggregator.save_cache(save_path)

    def load_user_cache(self, load_path):
        """加载用户缓存"""
        if self.neighbor_aggregator is not None:
            self.neighbor_aggregator.load_cache(load_path)

    def get_cache_stats(self):
        """获取缓存统计信息"""
        if self.neighbor_aggregator is not None:
            return self.neighbor_aggregator.get_cache_stats()
        return None
