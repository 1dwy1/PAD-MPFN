import numpy as np
from models.base.function import *

class ElementWiseMultiply(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, left, right):
        result = torch.mul(left, right)
        return result

class DotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, left, right):
        """

        Args:
            left: (batch_size, num_candidate, dim_embedding)
            right: (batch_size, dim_embedding)

        Returns:
            每个样本与候选新闻的预测值
            (shape): batch_size, num_candidate

        """
        result = torch.bmm(left, right.unsqueeze(dim=-1)).squeeze(dim=-1)
        return result


class AttentionPooling(nn.Module):
    """
    注意力池化层
    """
    def __init__(self, emb_size, hidden_size):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(emb_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def initialize(self):
        nn.init.xavier_uniform_(self.att_fc1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.att_fc1.bias)
        nn.init.xavier_uniform_(self.att_fc2.weight)
       
    def forward(self, x, attn_mask=None):
        e = self.att_fc1(x) # W_h · h_i^k
        e = nn.Tanh()(e) # tanh(W_h · h_i^k)
        alpha = self.att_fc2(e) # q_h^⊤ · tanh(W_h · h_i^k)
        alpha = torch.exp(alpha)  # exp(q_h^⊤ · tanh(W_h · h_i^k))

        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        # 归一化处理
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        # 加权求和
        x = torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        """
            Q: batch_size, n_head, candidate_num, d_k
            K: batch_size, n_head, candidate_num, d_k
            V: batch_size, n_head, candidate_num, d_v
            attn_mask: batch_size, n_head, candidate_num
            Return: batch_size, n_head, candidate_num, d_v
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)

        if attn_mask is not None:
            scores = scores * attn_mask.unsqueeze(dim=-2)

        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, head_num, head_dim, residual=False):
        super().__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        # residual：控制是否添加残差连接 如果启用，它会将输入直接加到输出上。
        self.residual = residual

        self.W_Q = nn.Linear(key_size, self.head_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(query_size, self.head_dim * self.head_num, bias=False)
        self.W_V = nn.Linear(value_size, self.head_dim * self.head_num, bias=True)

        self.scaled_dot_product_attn = ScaledDotProductAttention(self.head_dim)
        self.apply(xavier)

    def initialize(self):
        nn.init.zeros_(self.W_Q.bias)
        nn.init.zeros_(self.W_V.bias)


    def forward(self, Q, K, V, mask=None):
        """
            Q: batch_size, candidate_num, news_dim
            K: batch_size, candidate_num, news_dim
            V: batch_size, candidate_num, news_dim
            mask: batch_size, candidate_num
            Return: batch_size,  candidate_num, head_num * head_dim
        """
        batch_size = Q.shape[0]
        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.head_num, -1)

        q_s = self.W_Q(Q).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)

        context = self.scaled_dot_product_attn(q_s, k_s, v_s, mask)
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.head_dim)
        if self.residual:
            output += Q
        return output


