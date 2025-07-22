import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UserNeighborAggregator(nn.Module):
    """
    用户邻居聚合器 (NUPIE - Neighbor User Preference Information Enhancement)
    
    实现PAD-MPFN算法中的用户邻居聚合模块，通过聚合相似用户的兴趣信息
    来增强当前用户的表示。
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        self.k_neighbors = cfg.model.k_neighbors
        self.similarity_threshold = cfg.model.similarity_threshold
        self.aggregation_method = cfg.model.aggregation_method
        
        # 用户嵌入缓存
        self.user_embeddings_cache: Dict[str, torch.Tensor] = {}
        self.cache_size = cfg.model.cache_size
        self.cache_update_frequency = cfg.model.cache_update_frequency
        self.update_counter = 0
        
        # 邻居权重学习
        self.neighbor_weight_learner = nn.Sequential(
            nn.Linear(self.news_dim * 2, self.news_dim),
            nn.ReLU(),
            nn.Linear(self.news_dim, 1),
            nn.Sigmoid()
        )
        
        # 聚合后的特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.news_dim * 2, self.news_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.news_dim, self.news_dim)
        )
        
        logger.info(f"UserNeighborAggregator initialized with k={self.k_neighbors}, "
                   f"cache_size={self.cache_size}, method={self.aggregation_method}")
    
    def compute_user_similarity(self, user_emb1: torch.Tensor, user_emb2: torch.Tensor) -> torch.Tensor:
        """
        计算两个用户嵌入之间的相似度
        
        Args:
            user_emb1: 用户1的嵌入 [batch_size, news_dim]
            user_emb2: 用户2的嵌入 [batch_size, news_dim]
            
        Returns:
            相似度分数 [batch_size]
        """
        # 确保在同一设备上
        device = user_emb1.device
        user_emb2 = user_emb2.to(device)
        
        # 归一化嵌入
        user_emb1_norm = F.normalize(user_emb1, p=2, dim=1)
        user_emb2_norm = F.normalize(user_emb2, p=2, dim=1)
        
        # 计算余弦相似度
        similarity = torch.sum(user_emb1_norm * user_emb2_norm, dim=1)
        
        return similarity
    
    def find_top_k_neighbors(self, current_user_emb: torch.Tensor, 
                           user_ids: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
        """
        为每个用户找到top-k个最相似的邻居
        
        Args:
            current_user_emb: 当前用户的嵌入 [batch_size, news_dim]
            user_ids: 用户ID列表 [batch_size]
            
        Returns:
            neighbor_indices: 每个用户的邻居索引 [batch_size, k]
            neighbor_similarities: 每个用户的邻居相似度 [batch_size, k]
        """
        batch_size = current_user_emb.shape[0]
        device = current_user_emb.device
        
        neighbor_indices = []
        neighbor_similarities = []
        
        for i, user_id in enumerate(user_ids):
            user_emb = current_user_emb[i:i+1]  # [1, news_dim]
            
            # 获取缓存中的所有用户嵌入
            cached_embeddings = []
            cached_user_ids = []
            
            for cached_user_id, cached_emb in self.user_embeddings_cache.items():
                if cached_user_id != user_id:  # 排除自己
                    cached_embeddings.append(cached_emb)
                    cached_user_ids.append(cached_user_id)
            
            if not cached_embeddings:
                # 如果没有缓存的用户，返回空结果
                neighbor_indices.append([])
                neighbor_similarities.append([])
                continue
            
            # 将所有缓存的嵌入移到同一设备
            cached_embeddings = [emb.to(device) for emb in cached_embeddings]
            cached_embeddings_tensor = torch.stack(cached_embeddings)  # [num_cached, news_dim]
            
            # 计算相似度
            similarities = self.compute_user_similarity(
                user_emb.expand(cached_embeddings_tensor.shape[0], -1),
                cached_embeddings_tensor
            )  # [num_cached]
            
            # 找到top-k邻居
            if len(similarities) > 0:
                # 应用相似度阈值过滤
                valid_mask = similarities >= self.similarity_threshold
                if valid_mask.sum() > 0:
                    similarities = similarities[valid_mask]
                    valid_indices = torch.where(valid_mask)[0]
                    
                    # 选择top-k
                    k = min(self.k_neighbors, len(similarities))
                    top_k_values, top_k_indices = torch.topk(similarities, k)
                    
                    # 转换为原始索引
                    selected_indices = valid_indices[top_k_indices].cpu().tolist()
                    selected_similarities = top_k_values.cpu().tolist()
                else:
                    selected_indices = []
                    selected_similarities = []
            else:
                selected_indices = []
                selected_similarities = []
            
            neighbor_indices.append(selected_indices)
            neighbor_similarities.append(selected_similarities)
        
        return neighbor_indices, neighbor_similarities
    
    def aggregate_neighbor_embeddings(self, neighbor_embeddings: List[torch.Tensor], 
                                    neighbor_similarities: List[float]) -> torch.Tensor:
        """
        聚合邻居嵌入
        
        Args:
            neighbor_embeddings: 邻居嵌入列表
            neighbor_similarities: 邻居相似度列表
            
        Returns:
            聚合后的嵌入 [news_dim]
        """
        if not neighbor_embeddings:
            return torch.zeros(self.news_dim, device=neighbor_embeddings[0].device if neighbor_embeddings else 'cpu')
        
        # 确保所有嵌入在同一设备上
        device = neighbor_embeddings[0].device
        neighbor_embeddings = [emb.to(device) for emb in neighbor_embeddings]
        
        if self.aggregation_method == 'weighted_sum':
            # 加权求和
            weights = torch.tensor(neighbor_similarities, device=device)
            weights = F.softmax(weights, dim=0)
            
            neighbor_embeddings_tensor = torch.stack(neighbor_embeddings)  # [k, news_dim]
            aggregated = torch.sum(neighbor_embeddings_tensor * weights.unsqueeze(1), dim=0)
            
        elif self.aggregation_method == 'attention':
            # 注意力机制
            neighbor_embeddings_tensor = torch.stack(neighbor_embeddings)  # [k, news_dim]
            
            # 计算注意力权重
            attention_weights = F.softmax(
                torch.matmul(neighbor_embeddings_tensor, neighbor_embeddings_tensor.transpose(0, 1)),
                dim=1
            )
            
            aggregated = torch.sum(neighbor_embeddings_tensor * attention_weights.diagonal().unsqueeze(1), dim=0)
            
        elif self.aggregation_method == 'learned_weight':
            # 学习权重聚合
            neighbor_embeddings_tensor = torch.stack(neighbor_embeddings)  # [k, news_dim]
            
            # 使用学习的权重
            weights = []
            for i, emb in enumerate(neighbor_embeddings):
                # 将当前嵌入和邻居嵌入拼接
                combined = torch.cat([emb, neighbor_embeddings_tensor[i]], dim=0)
                weight = self.neighbor_weight_learner(combined.unsqueeze(0)).squeeze(0)
                weights.append(weight)
            
            weights = torch.cat(weights)  # [k]
            weights = F.softmax(weights, dim=0)
            
            aggregated = torch.sum(neighbor_embeddings_tensor * weights.unsqueeze(1), dim=0)
            
        else:
            # 默认：简单平均
            neighbor_embeddings_tensor = torch.stack(neighbor_embeddings)
            aggregated = torch.mean(neighbor_embeddings_tensor, dim=0)
        
        return aggregated
    
    def update_user_cache(self, user_ids: List[str], user_embeddings: torch.Tensor):
        """
        更新用户嵌入缓存
        
        Args:
            user_ids: 用户ID列表
            user_embeddings: 用户嵌入 [batch_size, news_dim]
        """
        self.update_counter += 1
        
        # 按频率更新缓存
        if self.update_counter % self.cache_update_frequency != 0:
            return
        
        device = user_embeddings.device
        
        for i, user_id in enumerate(user_ids):
            user_emb = user_embeddings[i].detach().cpu()  # 移到CPU并分离梯度
            
            # 更新或添加用户嵌入
            self.user_embeddings_cache[user_id] = user_emb
            
            # 如果缓存满了，删除最旧的条目
            if len(self.user_embeddings_cache) > self.cache_size:
                # 简单的LRU策略：删除第一个条目
                oldest_key = next(iter(self.user_embeddings_cache))
                del self.user_embeddings_cache[oldest_key]
    
    def forward(self, user_embeddings: torch.Tensor, user_ids: List[str]) -> torch.Tensor:
        """
        前向传播：聚合邻居信息增强用户表示
        
        Args:
            user_embeddings: 用户嵌入 [batch_size, news_dim]
            user_ids: 用户ID列表 [batch_size]
            
        Returns:
            增强后的用户嵌入 [batch_size, news_dim]
        """
        batch_size = user_embeddings.shape[0]
        device = user_embeddings.device
        
        # 更新用户缓存
        self.update_user_cache(user_ids, user_embeddings)
        
        # 找到top-k邻居
        neighbor_indices, neighbor_similarities = self.find_top_k_neighbors(user_embeddings, user_ids)
        
        enhanced_embeddings = []
        
        for i, user_id in enumerate(user_ids):
            user_emb = user_embeddings[i]  # [news_dim]
            
            # 获取邻居嵌入
            neighbor_emb_list = []
            neighbor_sim_list = []
            
            for j, neighbor_idx in enumerate(neighbor_indices[i]):
                if neighbor_idx < len(self.user_embeddings_cache):
                    cached_user_ids = list(self.user_embeddings_cache.keys())
                    neighbor_user_id = cached_user_ids[neighbor_idx]
                    neighbor_emb = self.user_embeddings_cache[neighbor_user_id].to(device)
                    neighbor_emb_list.append(neighbor_emb)
                    neighbor_sim_list.append(neighbor_similarities[i][j])
            
            # 聚合邻居嵌入
            if neighbor_emb_list:
                aggregated_neighbor_emb = self.aggregate_neighbor_embeddings(
                    neighbor_emb_list, neighbor_sim_list
                )
                
                # 特征融合：将原始用户嵌入和聚合的邻居嵌入融合
                combined_features = torch.cat([user_emb, aggregated_neighbor_emb], dim=0)
                enhanced_emb = self.feature_fusion(combined_features)
            else:
                # 如果没有邻居，保持原始嵌入
                enhanced_emb = user_emb
            
            enhanced_embeddings.append(enhanced_emb)
        
        # 堆叠所有增强的嵌入
        enhanced_embeddings = torch.stack(enhanced_embeddings)  # [batch_size, news_dim]
        
        return enhanced_embeddings
    
    def save_cache(self, save_path: str):
        """保存用户缓存到文件"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 将缓存转换为CPU张量
        cpu_cache = {}
        for user_id, emb in self.user_embeddings_cache.items():
            cpu_cache[user_id] = emb.cpu()
        
        with open(save_path, 'wb') as f:
            pickle.dump(cpu_cache, f)
        
        logger.info(f"User cache saved to {save_path} with {len(cpu_cache)} users")
    
    def load_cache(self, load_path: str):
        """从文件加载用户缓存"""
        load_path = Path(load_path)
        
        if not load_path.exists():
            logger.warning(f"Cache file {load_path} does not exist")
            return
        
        with open(load_path, 'rb') as f:
            self.user_embeddings_cache = pickle.load(f)
        
        logger.info(f"User cache loaded from {load_path} with {len(self.user_embeddings_cache)} users")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            'cache_size': len(self.user_embeddings_cache),
            'cache_usage': len(self.user_embeddings_cache) / self.cache_size if self.cache_size > 0 else 0,
            'update_counter': self.update_counter
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.user_embeddings_cache.clear()
        self.update_counter = 0
        logger.info("User cache cleared") 