# Bag-of-tricks-for-Transformers

## 快速开始

### 添加一个新 trick 实验

以添加 `my_trick` 为例，完整步骤：

**1. 创建方法文件夹，准备训练脚本**

```bash
mkdir -p exp/my_trick
cp exp/baseline-sp1024/train_gpt.py exp/my_trick/train_gpt.py
```

在 `exp/my_trick/train_gpt.py` 中修改你的 trick，改动处用注释标明：
```python
# trick: <简要描述你的改动>
```

其他部分尽量保持和 baseline 一致，保证可比性。

**2. 创建 manifest JSON**

复制 baseline 的 manifest 作为模板：
```bash
cp exp/baseline-sp1024/baseline-sp1024.json exp/my_trick/my_trick.json
```

修改 `exp/my_trick/my_trick.json`，把 `trainer_path` 和 `name` 改成你的方法：
```json
{
  "version": 1,
  "trainer_path": "exp/my_trick/train_gpt.py",
  "launcher": {
    "nproc_per_node": 8,
    "master_port_base": 29500
  },
  "defaults": {
    "data_path": "./data/datasets/fineweb10B_sp1024",
    "tokenizer_path": "./data/tokenizers/fineweb_1024_bpe.model",
    "vocab_size": 1024,
    "train_batch_tokens": 524288,
    "train_seq_len": 1024,
    "warmup_steps": 20,
    "warmdown_iters": 1200,
    "enable_wandb": 1,
    "wandb_mode": "online",
    "wandb_project": "bag-of-tricks-for-transformers",
    "num_layers": 9,
    "model_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 4,
    "mlp_mult": 2
  },
  "experiments": [
    {
      "name": "my_trick-fixed_time_10min",
      "trainer_path": "exp/my_trick/train_gpt.py",
      "control": {
        "mode": "fixed_compute",
        "target_wallclock_seconds": 600,
        "iterations_cap": 1000000
      }
    },
    {
      "name": "my_trick-fixed_tokens_10m",
      "trainer_path": "exp/my_trick/train_gpt.py",
      "control": {
        "mode": "fixed_tokens",
        "target_train_tokens": 10485760
      }
    }
  ]
}
```

**3. dry-run 检查配置**

```bash
uv run python exp/run_experiments.py exp/my_trick/my_trick.json --dry-run
```

确认模型参数、token 数、trainer 路径等都正确。

**4. 正式执行**

```bash
uv run python exp/run_experiments.py exp/my_trick/my_trick.json
```

8 卡 H100 上顺序执行 manifest 里的所有实验。

## 文件结构约定

每个方法一个文件夹，包含训练脚本和 manifest：
```
exp/
├── baseline-sp1024/
│   ├── train_gpt.py           # baseline 训练脚本
│   ├── baseline-sp1024.json   # baseline manifest
│   └── logs/                  # 训练输出（自动生成）
├── my_trick/
│   ├── train_gpt.py           # 你的 trick 训练脚本
│   ├── my_trick.json          # 你的 trick manifest
│   └── logs/                  # 训练输出（自动生成）
└── run_experiments.py         # 统一调度器
```

## 输出目录

默认输出到 manifest 所在文件夹的 `logs/` 子目录：
```
exp/my_trick/logs/
└── experiment-20260409-110120/          # batch_id（带时间戳）
    ├── my_trick-fixed_time_10min/       # 实验 1
    │   ├── log/<run_id>.txt
    │   ├── final_model.pt
    │   └── result.json
    ├── my_trick-fixed_tokens_10m/       # 实验 2
    │   ├── log/<run_id>.txt
    │   ├── final_model.pt
    │   └── result.json
    └── results.json                     # 批次汇总
```

可用 `--output-root` 覆盖输出位置。

## 控制模式

| 模式 | 说明 | manifest 必填字段 |
|------|------|-------------------|
| `fixed_compute` | 固定墙钟时间 | `target_wallclock_seconds` |
| `fixed_tokens` | 固定训练 token 数 | `target_train_tokens` |
| `fixed_model` | 固定模型，比较不同 token 预算 | `target_train_tokens` |

## result.json 里看什么

每个实验的 `result.json` 包含：
- `control.mode` / `control.target_train_tokens` / `control.actual_train_tokens`
- `control.target_wallclock_seconds` / `control.actual_wallclock_seconds`
- `metrics.final_val_bpb`
- `model.model_params`

拿它和 baseline 的 `result.json` 对比即可。

## 执行单次训练（不经过调度器）

```bash
WANDB_MODE=online \
WANDB_PROJECT=bag-of-tricks-for-transformers \
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
OUTPUT_DIR=./exp/baseline-sp1024/logs/manual-baseline-sp1024 \
uv run torchrun --standalone --nproc_per_node=8 ./exp/baseline-sp1024/train_gpt.py
```
