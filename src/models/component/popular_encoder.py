import torch
import torch.nn as nn


class PopularityEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.news_dim = cfg.model.head_dim * cfg.model.head_num
        self.user_dim = 400

    def forward(self, clicked_news_embeddings, popularity_scores):
        """
        增强版流行度编码器，可处理不同维度的输入

        Args:
            clicked_news_embeddings: (batch_size, H, news_dim) 或 (H, news_dim)
            popularity_scores: (batch_size, H) 或 (H,)

        Returns:
            user_popularity: (batch_size, news_dim) 或 (news_dim,)
        """

        # 统一处理输入维度
        if clicked_news_embeddings.dim() == 2:
            clicked_news_embeddings = clicked_news_embeddings.unsqueeze(0)  # (1, H, D)

        if popularity_scores.dim() == 1:
            popularity_scores = popularity_scores.unsqueeze(0)  # (1, H)

        # 确保长度匹配
        seq_len = clicked_news_embeddings.size(1)
        if popularity_scores.size(1) < seq_len:
            # 填充不足部分
            padding = torch.zeros(popularity_scores.size(0),
                                  seq_len - popularity_scores.size(1),
                                  device=popularity_scores.device)
            popularity_scores = torch.cat([popularity_scores, padding], dim=1)
        elif popularity_scores.size(1) > seq_len:
            # 截取多余部分
            popularity_scores = popularity_scores[:, :seq_len]

        # 计算加权平均
        weighted_sum = torch.sum(
            clicked_news_embeddings * popularity_scores.unsqueeze(-1),
            dim=1
        )

        sum_weights = torch.sum(popularity_scores, dim=1, keepdim=True) + 1e-8

        # 恢复原始输出维度
        if clicked_news_embeddings.dim() == 2 and popularity_scores.dim() == 1:
            return (weighted_sum / sum_weights).squeeze(0)
        return weighted_sum / sum_weights