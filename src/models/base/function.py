import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(X, valid_lens):
    # 输入X是一个3维张量(batch_size, num_queries, num_keys),valid_lens:一个1D或者2D的张量对应样本有效长度
    """X是一个注意力分数矩阵，valid_lens是每个序列的有效长度。masked_softmax的作用是：
        只对每个序列的有效部分（由 valid_lens 决定）进行 softmax计算
        对超过有效长度的无效元素，使用掩码将其排除在 softmax 计算之外`"""

    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e4)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def xavier(m):
    """
    Xavier 对于每一层的权重初始化，避免在深层网络中出现梯度消失或梯度爆炸的问题。
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_normal(m):
    """
    正态分布初始化
    """
    if type(m) == nn.Linear:
        # 权重初始化为均值0，标准差0.01的正态分布
        nn.init.normal_(m.weight, mean=0, std=0.01)
        #  偏置初始化为0
        nn.init.zeros_(m.bias)


def init_constant(m):
    """
    初始化权重，初始化偏置
    """
    if type(m) == nn.Linear:
        #   权重初始化为1
        nn.init.constant_(m.weight, 1)
        #  偏置初始化为0
        nn.init.zeros_(m.bias)


if __name__ == '__main__':
    # X的形状(2,2,3)每个样本都有 2 个查询，每个查询与 3 个键相关联。
    X = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                      [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
    valid_lens = torch.tensor([2, 3])
    result = masked_softmax(X, valid_lens)
    print(result)

