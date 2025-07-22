import torch
import torch.nn as nn
import torch.nn.functional as F



class NCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, score, label):
        """

        Args:
            score: (batch_size, candidate_num)
            label: (batch_size, candidate_num)

        Returns:
            loss: 负对数损失

        """
        # (batch_size)
        result = F.log_softmax(score, dim=1)
        loss = F.nll_loss(result, label)
        return loss


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, score, label):
        """
        Args:
            score: (batch_size, 2) - 模型输出的 logits（未经 softmax 归一化）
            label: (batch_size) - 正确的标签（0 或 1）

        Returns:
            loss: 交叉熵损失
        """
        # 使用 CrossEntropyLoss 计算损失
        criterion = nn.CrossEntropyLoss()
        loss = criterion(score, label)
        return loss