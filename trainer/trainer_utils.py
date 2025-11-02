"""
训练工具函数集合
"""
import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


# def get_lr(current_step, total_steps, lr):
#     return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))
def get_lr(global_step, total_updates, base_lr, warmup_ratio=0.03, min_lr_ratio=0.1):
    warmup = int(total_updates * warmup_ratio)
    min_lr = base_lr * min_lr_ratio
    if global_step < warmup:
        return base_lr * (global_step + 1) / max(1, warmup)  # 线性升
    # 余弦退火到 min_lr
    progress = (global_step - warmup) / max(1, total_updates - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../SmolMind', save_dir='../out', device='cuda'):
    from transformers import AutoTokenizer
    from model.SmolMind import SmolMindForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = SmolMindForCausalLM(lm_config)
    
    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    
    Logger(f'所加载Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


def fix_arg_types(args):
    """
    自动修复 argparse / YAML 合并后类型错误的问题。
    将数字字符串自动转为 int / float，布尔字符串转为 bool。
    """
    import re

    numeric_keys = [
        "learning_rate", "grad_clip", "epochs", "batch_size",
        "num_workers", "accumulation_steps", "log_interval",
        "save_interval", "hidden_size", "num_hidden_layers",
        "max_seq_len"
    ]

    bool_true = {"true", "yes", "1", "on"}
    bool_false = {"false", "no", "0", "off"}

    for key, val in vars(args).items():
        # 只处理字符串类型
        if isinstance(val, str):
            v = val.strip().lower()

            # ✅ 自动识别布尔值
            if v in bool_true:
                setattr(args, key, True)
            elif v in bool_false:
                setattr(args, key, False)

            # ✅ 自动识别数字（支持科学计数法、负号、小数）
            elif re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", v):
                try:
                    if any(c in v for c in [".", "e", "E", "-"]):
                        setattr(args, key, float(val))
                    else:
                        setattr(args, key, int(val))
                except ValueError:
                    pass

    return args
