import os
import sys
import argparse
import time
import warnings
import torch
import torch.distributed as dist
import yaml
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# ========= 导入自定义模块 =========
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.SmolMind import SmolMindConfig
from dataset.lm_dataset import PretrainDataset, SFTDataset
from trainer.trainer_utils import *

warnings.filterwarnings('ignore')


# ========== 配置加载器 ==========
def load_config(config_path):
    """从 YAML 或 JSON 加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError("配置文件仅支持 .yaml / .json 格式")


# ========== 参数定义 + 合并配置 ==========
def parse_args():
    parser = argparse.ArgumentParser(description="SmolMind Full SFT")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")

    # 默认参数（可以被配置文件或命令行覆盖）
    default_args = {
        "save_dir": "../out",
        "save_weight": "full_sft(512)",
        "epochs": 4,
        "batch_size": 100,
        "learning_rate": 4e-7,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": "bfloat16",
        "num_workers": 8,
        "accumulation_steps": 4,
        "grad_clip": 1.0,
        "log_interval": 100,
        "save_interval": 200,
        "hidden_size": 768,
        "num_hidden_layers": 16,
        "max_seq_len": 1024,
        "use_moe": False,
        "data_path": "../dataset/sft_mini_1024.jsonl",
        "from_weight": "pretrain",
        "from_resume": 0,
        "use_wandb": False,
        "wandb_project": "SmolMind-Full-SFT"
    }

    # 将默认参数全部注册
    for k, v in default_args.items():
        arg_type = type(v) if v is not None else str
        parser.add_argument(f"--{k}", type=arg_type, default=v)

    args = parser.parse_args()

    # 如果指定了配置文件，则加载并覆盖默认参数
    if args.config:
        cfg = load_config(args.config)
        print(f"✅ 从配置文件加载参数: {args.config}")
        for k, v in cfg.items():
            setattr(args, k, v)

    return fix_arg_types(args)

# ========== 训练循环 ==========
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    progress_bar = tqdm(
        enumerate(loader, start=start_step + 1),
        total=iters,
        ncols=120,
        desc=f"Epoch [{epoch + 1}/{args.epochs}]"
    )

    for step, (X, Y, loss_mask) in progress_bar:
        X, Y, loss_mask = X.to(args.device), Y.to(args.device), loss_mask.to(args.device)

        # 学习率调度
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for p in optimizer.param_groups:
            p["lr"] = lr

        with autocast_ctx:
            res = model(X)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志与进度条
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}', 'lr': f'{current_lr:.2e}', 'eta(min)': f'{eta_min:.1f}'})
            Logger(f"Epoch[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} ETA:{eta_min}min")
            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # 定期保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            print(f"✅ Step {step} 模型已保存: {ckp}")
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir="../checkpoints")
            model.train()

    # 每轮结束保存
    if is_main_process():
        model.eval()
        moe_suffix = "_moe" if lm_config.use_moe else ""
        ckp = f"../checkpoints/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}_epoch{epoch + 1}.pth"
        torch.save({k: v.half() for k, v in model.state_dict().items()}, ckp)
        print(f"✅ Epoch {epoch + 1} 权重保存: {ckp}")
        model.train()


# ========== 主入口 ==========
if __name__ == "__main__":
    args = parse_args()
    # 初始化
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 模型与环境
    lm_config = SmolMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type="cuda", dtype=dtype)

    # wandb
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb.init(
            project=args.wandb_project,
            name=f"SmolMind-Full-SFT-Epoch{args.epochs}-Batch{args.batch_size}-LR{args.learning_rate}",
            id=wandb_id, resume=resume, config=lm_config.to_dict()
        )

    # 模型与数据
    model, tokenizer = init_model(lm_config, args.from_weight)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(device="cuda", enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 恢复训练状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch, start_step = ckp_data["epoch"], ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========= 正式训练 =========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else: # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)