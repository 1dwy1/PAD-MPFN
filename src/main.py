import os.path

import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.cuda import amp
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import wandb
from dataload.data_load import load_data
from dataload.data_preprocess import prepare_preprocessed_data
from src.models.utils.common import *
from src.models.utils.metrics import *

os.environ["WANDB_API_KEY"] = "a89657e7cfa75b828fc2dba08965b7c88ec2a0f8"
os.environ["WANDB_MODE"] = "offline"


def train(model, optimizer, scaler, scheduler, dataloader, local_rank, cfg, early_stopping):
    model.train()
    torch.set_grad_enabled(True)

    sum_loss = torch.zeros(1).to(local_rank)
    sum_auc = torch.zeros(1).to(local_rank)

    for cnt, batch_data in enumerate(tqdm(dataloader,
                                        total=int(cfg.num_epochs * (
                                                cfg.dataset.pos_count // cfg.batch_size + 1)),
                                        desc=f"[{local_rank}] Training"), start=1):
        
        # 处理不同长度的batch_data
        if len(batch_data) == 9:  # 包含user_ids
            (subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, labels, pop_value,
             candidate_click_counts, user_ids) = batch_data
        elif len(batch_data) == 8:  # 不包含user_ids
            (subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, labels, pop_value,
             candidate_click_counts) = batch_data
            user_ids = None
        else:
            raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
        
        subgraph = subgraph.to(local_rank, non_blocking=True)
        mapping_idx = mapping_idx.to(local_rank, non_blocking=True)
        candidate_news = candidate_news.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)
        pop_value = pop_value.to(local_rank, non_blocking=True)

        with open(cfg.path.log_dir, 'a') as f:
            f.write(f"label: {labels}\n")

        with amp.autocast():
            bz_loss, y_hat = model(subgraph, mapping_idx, candidate_news, pop_value, labels, user_ids)

        scaler.scale(bz_loss).backward()
        if cnt % cfg.accumulation_steps == 0 or cnt == int(cfg.dataset.pos_count / cfg.batch_size):
            scaler.step(optimizer)
            old_scaler = scaler.get_scale()
            scaler.update()
            new_scaler = scaler.get_scale()
            if new_scaler >= old_scaler:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        sum_loss += bz_loss.data.float()
        sum_auc += area_under_curve(labels, y_hat)

        if cnt % cfg.log_steps == 0:
            if local_rank == 0:
                cache_stats = model.module.get_cache_stats()
                if cache_stats:
                    wandb.log({
                        "train_loss": sum_loss.item() / cfg.log_steps, 
                        "train_auc": sum_auc.item() / cfg.log_steps,
                        "user_cache_size": cache_stats['cache_size'],
                        "user_cache_usage": cache_stats['cache_usage']
                    })
                else:
                    wandb.log({"train_loss": sum_loss.item() / cfg.log_steps, "train_auc": sum_auc.item() / cfg.log_steps})
            print('[{}] Ed: {}, average_loss: {:.5f}, average_acc: {:.5f}'.format(
                local_rank, cnt * cfg.batch_size, sum_loss.item() / cfg.log_steps, sum_auc.item() / cfg.log_steps))
            sum_loss.zero_()
            sum_auc.zero_()

        if cnt > int(cfg.val_skip_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)) and cnt % cfg.val_steps == 0:
            res = val(model, local_rank, cfg)
            model.train()

            if local_rank == 0:
                pretty_print(res)
                wandb.log(res)

            early_stop, get_better = early_stopping(res['auc'])
            if early_stop:
                print("Early Stop.")
                break
            elif get_better:
                print(f"Better Result!")
                if local_rank == 0:
                    save_model(cfg, model, optimizer, f"{cfg.ml_label}_auc{res['auc']}")
                    # 保存用户缓存
                    cache_path = Path(cfg.path.ckp_dir) / f"user_cache_{cfg.ml_label}_auc{res['auc']}.pkl"
                    model.module.save_user_cache(cache_path)
                    wandb.run.summary.update({"best_auc": res["auc"], "best_mrr": res['mrr'],
                                              "best_ndcg5": res['ndcg5'], "best_ndcg10": res['ndcg10']})


def val(model, local_rank, cfg):
    model.eval()
    dataloader = load_data(cfg, mode='val', model=model, local_rank=local_rank)
    tasks = []
    with torch.no_grad():
        for cnt, batch_data in enumerate(tqdm(dataloader,
                                            total=int(cfg.dataset.val_len / cfg.gpu_num),
                                            desc=f"[{local_rank}] Validating")):
            
            # 处理不同长度的batch_data
            if len(batch_data) == 9:  # 包含user_id
                subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, news_p, labels, user_id = batch_data
            elif len(batch_data) == 8:  # 不包含user_id
                subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, news_p, labels = batch_data
                user_id = None
            else:
                raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
            
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(local_rank, non_blocking=True)

            scores = model.module.validation_process(subgraph, mappings, candidate_emb, news_p, [user_id] if user_id else None)

            tasks.append((labels.tolist(), scores))

    with mp.Pool(processes=cfg.num_workers) as pool:
        results = pool.map(cal_metric, tasks)
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = np.array(results).T

    # barrier
    torch.distributed.barrier()

    reduced_auc = reduce_mean(torch.tensor(np.nanmean(val_auc)).float().to(local_rank), cfg.gpu_num)
    reduced_mrr = reduce_mean(torch.tensor(np.nanmean(val_mrr)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg5 = reduce_mean(torch.tensor(np.nanmean(val_ndcg5)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg10 = reduce_mean(torch.tensor(np.nanmean(val_ndcg10)).float().to(local_rank), cfg.gpu_num)

    res = {
        "auc": reduced_auc.item(),
        "mrr": reduced_mrr.item(),
        "ndcg5": reduced_ndcg5.item(),
        "ndcg10": reduced_ndcg10.item(),
    }

    return res


def main_worker(local_rank, cfg):
    # -----------------------------------------Environment Initial
    seed_everything(cfg.seed)
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=cfg.gpu_num,
                            rank=local_rank)

    # -----------------------------------------Dataset & Model Load
    num_training_steps = int(cfg.num_epochs * cfg.dataset.pos_count / (cfg.batch_size * cfg.accumulation_steps))
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio + 1)
    train_dataloader = load_data(cfg, mode='train', local_rank=local_rank)
    model = load_model(cfg).to(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    lr_lambda = lambda step: 1.0 if step > num_warmup_steps else step / num_warmup_steps
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------Load Checkpoint & optimizer
    if cfg.load_checkpoint:
        file_path = Path(f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_{cfg.load_mark}.pth")
        checkpoint = torch.load(file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])  # After Distributed
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 尝试加载用户缓存
        cache_path = Path(f"{cfg.path.ckp_dir}/user_cache_{cfg.load_mark}.pkl")
        if cache_path.exists():
            model.load_user_cache(cache_path)
            print(f"Loaded user cache from {cache_path}")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer.zero_grad(set_to_none=True)
    scaler = amp.GradScaler()

    # ------------------------------------------Main Start
    early_stopping = EarlyStopping(cfg.early_stop_patience)

    if local_rank == 0:
        wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                   project=cfg.logger.exp_name, name=cfg.logger.run_name)
        print(model)

    train(model, optimizer, scaler, scheduler, train_dataloader, local_rank, cfg, early_stopping)
    #val(model, local_rank, cfg)
    if local_rank == 0:
        wandb.finish()


@hydra.main(version_base="1.2", config_path=os.path.join(get_root(), "configs"), config_name="small")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    cfg.gpu_num = torch.cuda.device_count()
    prepare_preprocessed_data(cfg)
    mp.spawn(main_worker, nprocs=cfg.gpu_num, args=(cfg,))


if __name__ == "__main__":
    main()
