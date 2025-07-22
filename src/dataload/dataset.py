import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import numpy as np


class TrainDataset(IterableDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg):
        super().__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = cfg.gpu_num

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            # 如果需要在前面填充
            pad_x = torch.cat([torch.full((fix_length - len(x),), padding_value, dtype=x.dtype), x[-fix_length:]])
            mask = torch.cat([torch.zeros(fix_length - len(x)), torch.ones(min(fix_length, len(x)))])
        else:
            # 如果需要在后面填充
            pad_x = torch.cat([x[-fix_length:], torch.full((fix_length - len(x),), padding_value, dtype=x.dtype)])
            mask = torch.cat([torch.ones(min(fix_length, len(x))), torch.zeros(fix_length - len(x))])
        return pad_x, mask

    def line_mapper(self, line):

        line = line.strip().split('\t')
        user_id = line[1]  # 提取用户ID
        click_id = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        clicked_index, clicked_mask = self.pad_to_fix_len(self.trans_to_nindex(click_id), self.cfg.model.his_size)
        clicked_input = self.news_input[clicked_index]

        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        return clicked_input, clicked_mask, candidate_input, label, user_id

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)


class TrainGraphDataset(TrainDataset):
    def __init__(self, filename, news_index, news_input, news_p, local_rank, cfg,
                 neighbor_dict, news_graph, entity_neighbors):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)
        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors
        self.news_p = news_p

        self.news_p_value = torch.zeros(len(news_index) + 1, dtype=torch.float32)
        for news_id, value in news_p.items():
            if news_id in news_index:
                self.news_p_value[news_index[news_id]] = value

    def line_mapper(self, line, sum_num_news):
        line = line.strip().split('\t')
        user_id = line[1]  # 提取用户ID
        click_id = line[3].split()[-self.cfg.model.his_size:]
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        # 处理点击新闻
        click_idx = self.trans_to_nindex(click_id)
        click_p = self.news_p_value[click_idx]

        # 确保click_counts长度固定
        if len(click_p) < self.cfg.model.his_size:
            click_p = F.pad(click_p, (0, self.cfg.model.his_size - len(click_p)), mode='constant',
                                 value=0)
        else:
            click_p = click_p[:self.cfg.model.his_size]

        # 构建子图
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), sum_num_news)
        padded_mapping_idx = F.pad(mapping_idx, (self.cfg.model.his_size - len(mapping_idx), 0), mode='constant',
                                   value=-1)

        # 处理候选新闻
        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]
        candidate_p_counts = self.news_p_value[sample_news]

        # 处理实体
        if self.cfg.model.use_entity:
            origin_entity = candidate_input[:, -3 - self.cfg.model.entity_size:-3]
            candidate_neighbor_entity = np.zeros(
                (len(sample_news) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)

            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                valid_len = min(len(self.entity_neighbors[idx]), self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(sample_news), -1)
            entity_mask = (candidate_neighbor_entity > 0).astype(np.float32)
            candidate_entity = np.concatenate([origin_entity, candidate_neighbor_entity], axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        return (sub_news_graph, padded_mapping_idx, candidate_input, candidate_entity,
                entity_mask, label, click_p, candidate_p_counts, sum_num_news + sub_news_graph.num_nodes, user_id)

    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device

        if not subset:
            subset = [0]

        subset = torch.tensor(subset, dtype=torch.long, device=device)

        unique_subset, unique_mapping = torch.unique(subset, sorted=True, return_inverse=True)
        subemb = self.news_graph.x[unique_subset]

        sub_edge_index, sub_edge_attr = subgraph(unique_subset, self.news_graph.edge_index, self.news_graph.edge_attr,
                                                 relabel_nodes=True, num_nodes=self.news_graph.num_nodes)

        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return sub_news_graph, unique_mapping[:k] + sum_num_nodes

    def __iter__(self):
        while True:
            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []
            candidate_entity_list = []
            entity_mask_list = []
            click_p_list = []
            candidate_p_list = []
            user_ids = []

            sum_num_news = 0
            with open(self.filename) as f:
                for line in f:
                    (sub_newsgraph, padded_mapping_idx, candidate_input, candidate_entity,
                     entity_mask, label, click_p, candidate_p_counts, sum_num_news, user_id) = self.line_mapper(line,
                                                                                                                sum_num_news)

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)
                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))
                    click_p_list.append(click_p)
                    candidate_p_list.append(candidate_p_counts)
                    user_ids.append(user_id)

                    if len(clicked_graphs) == self.batch_size:
                        batch = Batch.from_data_list(clicked_graphs)
                        candidates = torch.stack(candidates)
                        mappings = torch.stack(mappings)
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)
                        click_p_list = torch.stack(click_p_list)
                        candidate_p_list = torch.stack(candidate_p_list)
                        labels = torch.tensor(labels, dtype=torch.long)

                        yield (batch, mappings, candidates, candidate_entity_list,
                               entity_mask_list, labels, click_p_list, candidate_p_list, user_ids)

                        clicked_graphs, mappings, candidates, labels = [], [], [], []
                        candidate_entity_list, entity_mask_list = [], []
                        click_p_list, candidate_p_list = [], []
                        user_ids = []
                        sum_num_news = 0

            if len(clicked_graphs) > 0:
                batch = Batch.from_data_list(clicked_graphs)
                candidates = torch.stack(candidates)
                mappings = torch.stack(mappings)
                candidate_entity_list = torch.stack(candidate_entity_list)
                entity_mask_list = torch.stack(entity_mask_list)
                click_p_list = torch.stack(click_p_list)
                candidate_p_list = torch.stack(candidate_p_list)
                labels = torch.tensor(labels, dtype=torch.long)

                yield (batch, mappings, candidates, candidate_entity_list,
                       entity_mask_list, labels, click_p_list, candidate_p_list, user_ids)


class ValidGraphDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input, news_p, local_rank, cfg,
                 neighbor_dict, news_graph, entity_neighbors, news_entity):
        super().__init__(filename, news_index, news_input, news_p, local_rank, cfg,
                        neighbor_dict, news_graph, entity_neighbors)
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity
        self.news_p = news_p
        self.news_p_value = torch.zeros(len(news_index) + 1, dtype=torch.float32)
        for news_id, value in news_p.items():
            if news_id in news_index:
                self.news_p_value[news_index[news_id]] = value


    def line_mapper(self, line):
        line = line.strip().split('\t')
        user_id = line[1]  # 提取用户ID
        click_id = line[3].split()[-self.cfg.model.his_size:]

        click_idx = self.trans_to_nindex(click_id)

        clicked_entity = self.news_entity[click_idx]

        click_p = self.news_p_value[click_idx]

        click_p = click_p[:self.cfg.model.his_size]
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

        # ------------------ Entity --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]

        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]
            candidate_neighbor_entity = np.zeros(
                (len(candidate_index) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index),
                                                                          self.cfg.model.entity_size * self.cfg.model.entity_neighbors)

            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1

            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        batch = Batch.from_data_list([sub_news_graph])

        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, click_p, labels, user_id

    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:
                batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, click_p, labels, user_id = self.line_mapper(
                    line)
                yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, click_p, labels, user_id


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)