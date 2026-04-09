# 统一实验调度说明

## 目的

现在仓库的目标不是单独做 scaling law sweep，而是围绕同一套 baseline 设置做统一实验调度。

典型用法是：
- baseline 先单独跑一次，得到基准结果
- 新增一个 trick 时，在 `exp/<method_name>/train_gpt.py` 中实现该方法
- 再为该方法写一个对应 manifest
- 用统一调度器在相同控制条件下执行，例如：
  - 固定训练时间（当前按固定墙钟时间实现）
  - 固定训练 token 数
- 最后拿该方法的 `result.json` 与 baseline 的 `result.json` 做对比

核心目标有三个：
1. 用统一 manifest 描述一组实验，而不是手工改环境变量反复执行
2. 对不同方法复用同一套控制维度，保证 baseline 与 trick 的可比性
3. 每个 run 自动输出结构化 `result.json`，用于比较 final bpb 和实际参数量

---

## 当前代码结构

### 1. 统一调度器

文件：`exp/run_experiments.py`

职责：
- 读取 manifest JSON
- 校验公共训练配置是否合法
- 根据 control mode 展开最终训练配置
- 解析每条 experiment 的 `trainer_path`
- 顺序执行 8 卡训练
- 收集每个 run 的 `result.json`
- 在 batch 结束后生成批次级别的 `results.json`

当前命令形式：

```bash
uv run python exp/run_experiments.py exp/baseline-sp1024/baseline-sp1024.json
```

也支持 dry-run：

```bash
uv run python exp/run_experiments.py exp/baseline-sp1024/baseline-sp1024.json --dry-run
```

### 2. baseline trainer

当前 baseline trainer：`exp/baseline-sp1024/train_gpt.py`

这个 trainer 在训练结束后会自动写出：

- `OUTPUT_DIR/result.json`

其中包含：
- 实验身份信息
  - `run_id`
  - `trainer`
  - `experiment_name`
- 控制信息
  - `control.mode`
  - `control.target_train_tokens`
  - `control.actual_train_tokens`
  - `control.target_wallclock_seconds`
  - `control.actual_wallclock_seconds`
  - `control.stopped_early`
- 模型信息
  - `model.num_layers`
  - `model.model_dim`
  - `model.num_heads`
  - `model.num_kv_heads`
  - `model.mlp_mult`
  - `model.vocab_size`
  - `model.model_params`
- 最终指标
  - `metrics.final_val_loss`
  - `metrics.final_val_bpb`
  - `metrics.final_eval_time_ms`
- 训练信息
  - `training.iterations`
  - `training.final_step`
  - `training.training_time_ms`
- 产物路径
  - `artifacts.model_path`
  - `artifacts.log_path`

`model.model_params` 会记录该方法的实际参数量，所以如果某个 trick 改大了参数量，也能直接从结果里看到。

---

## manifest 设计

推荐每个方法各自放一个 manifest，例如：

- `exp/baseline-sp1024/baseline-sp1024.json`
- `exp/my_trick/my_trick.json`

也就是说：
- baseline 有自己的 manifest
- 每个 trick 也有自己的 manifest
- 后续新增方法时，通常只需要改：
  - `trainer_path`
  - `experiments[].name`

当前 manifest 主结构示例：

```json
{
  "version": 1,
  "trainer_path": "exp/baseline-sp1024/train_gpt.py",
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
      "name": "baseline-sp1024-fixed_time_10min",
      "trainer_path": "exp/baseline-sp1024/train_gpt.py",
      "control": {
        "mode": "fixed_compute",
        "target_wallclock_seconds": 600,
        "iterations_cap": 1000000
      }
    },
    {
      "name": "baseline-sp1024-fixed_tokens_10m",
      "trainer_path": "exp/baseline-sp1024/train_gpt.py",
      "control": {
        "mode": "fixed_tokens",
        "target_train_tokens": 10485760
      }
    }
  ]
}
```

如果要加新 trick，可以复制 baseline manifest，再把：
- 顶层 `trainer_path`
- 每个 experiment 里的 `trainer_path`
- 每个 experiment 的 `name`

改成该 trick 自己的目录。

---

## 当前支持的控制模式

### 1. `fixed_compute`

当前语义：固定墙钟时间。

manifest 中填写：
- `target_wallclock_seconds`
- 通常还会配一个足够大的 `iterations_cap`

调度器会把它映射成：
- `MAX_WALLCLOCK_SECONDS=<target_wallclock_seconds>`
- `ITERATIONS=<iterations_cap>`

实际停止条件由 trainer 内部的墙钟时间控制。

适合做：
- baseline 与 trick 在同一台 8xH100 上、同样训练时间预算下的最终 bpb 对比

### 2. `fixed_tokens`

当前语义：固定总训练 token 数。

manifest 中填写：
- `target_train_tokens`

调度器会自动换算：
- `ITERATIONS = ceil(target_train_tokens / TRAIN_BATCH_TOKENS)`

并在 `result.json` 中同时记录：
- `control.target_train_tokens`
- `control.actual_train_tokens`

适合做：
- baseline 与 trick 在相同 token 预算下的最终 bpb 对比

### 3. `fixed_model`

当前仍保留兼容。

它和 `fixed_tokens` 一样通过 `target_train_tokens` 推导 `ITERATIONS`，只是语义上更偏向：
- 固定模型结构
- 比较不同 token 预算

当前仓库的主使用场景仍然是：
- `fixed_compute`
- `fixed_tokens`

---

## 运行方式

### 1. baseline

baseline 配置文件：`exp/baseline-sp1024/baseline-sp1024.json`

dry-run：

```bash
uv run python exp/run_experiments.py exp/baseline-sp1024/baseline-sp1024.json --dry-run
```

正式执行：

```bash
uv run python exp/run_experiments.py exp/baseline-sp1024/baseline-sp1024.json
```

### 2. 新 trick

以 `my_trick` 为例：

1. 创建目录并复制 baseline trainer：

```bash
mkdir -p exp/my_trick
cp exp/baseline-sp1024/train_gpt.py exp/my_trick/train_gpt.py
```

2. 修改 `exp/my_trick/train_gpt.py`
   - 改动处建议加注释，例如：

```python
# trick: add xxx
```

3. 复制 manifest：

```bash
cp exp/baseline-sp1024/baseline-sp1024.json exp/my_trick/my_trick.json
```

4. 修改 `exp/my_trick/my_trick.json`
   - `trainer_path` 改为 `exp/my_trick/train_gpt.py`
   - experiment 名称改成 `my_trick-*`

5. 执行：

```bash
uv run python exp/run_experiments.py exp/my_trick/my_trick.json --dry-run
uv run python exp/run_experiments.py exp/my_trick/my_trick.json
```

---

## 输出目录规则

默认情况下，调度器会把输出写到：

- `<manifest_dir>/logs/<batch_id>/`

例如 baseline：

- manifest: `exp/baseline-sp1024/baseline-sp1024.json`
- 默认输出根目录：`exp/baseline-sp1024/logs/`

某次 batch 结果会像这样：

```text
exp/baseline-sp1024/logs/
└── experiment-20260409-110120/
    ├── baseline-sp1024-fixed_time_10min/
    │   ├── log/
    │   │   └── <run_id>.txt
    │   ├── final_model.pt
    │   └── result.json
    ├── baseline-sp1024-fixed_tokens_10m/
    │   ├── log/
    │   ├── final_model.pt
    │   └── result.json
    └── results.json
```

如果需要，也可以通过：

```bash
--output-root <path>
```

覆盖默认输出路径。

---

## 当前代码检查结论

已核对当前代码与 README 表达的语义，结论如下：

### 已对齐的部分

1. 调度器已经是统一实验调度器，不再绑定 scaling baseline 语义
2. 支持每个方法各自维护自己的 manifest
3. 默认输出目录已经是 `manifest_dir/logs`
4. `exp/baseline-sp1024/train_gpt.py` 会写 `result.json`
5. `result.json` 已包含 `metrics.final_val_bpb` 和 `model.model_params`
6. `fixed_compute` 与 `fixed_tokens` 都能正确展开
