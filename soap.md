# SOAP Trick 适配文档

> 在 `Bag-soap` 内新增 `exp/soap/` trick：用 [SOAP 优化器](https://arxiv.org/abs/2409.11321) 替换 baseline 的 Muon。SOAP 算法**逐字**取自 [`nikhilvyas/modded-nanogpt-SOAP/soap.py`](https://github.com/nikhilvyas/modded-nanogpt-SOAP/blob/master/soap.py)；trainer 集成方式对齐参考 trick `Bag-of-tricks-for-Transformers/exp/muon-ema-update-smoothing/`。
>
> **默认运行配置**：单机 8 卡（沿用 baseline manifest 的 `nproc_per_node: 8`）。

---

## 项目约定（来自 `muon-ema-update-smoothing/`）

- `exp/run_experiments.py` 的 `ENV_KEY_MAP` **不扩展**；trick 超参走 `os.environ.get`。
- 新优化器 class **内联到 `train_gpt.py`**，**不新建** `.py` 辅助文件。
- Manifest `defaults` **不放** trick 超参；只含 baseline 通用参数。
- 每处改动打 `# trick:` 注释。
- 改动只发生在 optimizer 片段：hyperparams、optimizer class、optimizer 实例化、训练 loop 优化器调度段、W&B 日志。模型 / 数据 / eval 零改动。

---

## 新建文件清单（零改既有文件）

| # | 文件 | 来源 / 动作 |
|---|------|-------------|
| 1 | `exp/soap/train_gpt.py` | `cp exp/baseline-sp1024/train_gpt.py` 后做 6 处精确 diff（§A） |
| 2 | `exp/soap/soap.json` | `cp exp/baseline-sp1024/baseline-sp1024.json` 后改 `trainer_path` / `name`（§B） |
| 3 | `exp/soap/README.md` | 按 §C 模板新建 |
| 4 | `exp/soap/run.sh` | 2 行启动脚本（§D） |

---

## §A `exp/soap/train_gpt.py` 的 6 处精确 diff

### A1. 顶部 import（baseline 现有 import 块末尾追加）

```python
# trick: needed by inlined SOAP optimizer (sourced from nikhilvyas/modded-nanogpt-SOAP).
from itertools import chain
```

### A2. Hyperparameters 数据类（baseline 第 76-90 行，紧跟 `muon_*` 字段）

```python
# trick: SOAP hyperparameters; defaults mirror nikhilvyas/modded-nanogpt-SOAP/soap.py constructor verbatim.
soap_lr = float(os.environ.get("SOAP_LR", 3e-3))
soap_beta1 = float(os.environ.get("SOAP_BETA1", 0.95))
soap_beta2 = float(os.environ.get("SOAP_BETA2", 0.95))
soap_shampoo_beta = float(os.environ.get("SOAP_SHAMPOO_BETA", -1.0))
soap_eps = float(os.environ.get("SOAP_EPS", 1e-8))
soap_weight_decay = float(os.environ.get("SOAP_WEIGHT_DECAY", 0.01))
soap_precondition_frequency = int(os.environ.get("SOAP_PRECONDITION_FREQUENCY", 10))
soap_max_precond_dim = int(os.environ.get("SOAP_MAX_PRECOND_DIM", 10000))
soap_merge_dims = bool(int(os.environ.get("SOAP_MERGE_DIMS", "0")))
soap_precondition_1d = bool(int(os.environ.get("SOAP_PRECONDITION_1D", "0")))
soap_normalize_grads = bool(int(os.environ.get("SOAP_NORMALIZE_GRADS", "0")))
soap_data_format = os.environ.get("SOAP_DATA_FORMAT", "channels_first")
soap_correct_bias = bool(int(os.environ.get("SOAP_CORRECT_BIAS", "1")))
```

`muon_*` 字段保留不删（保持最小 diff，与参考 trick 习惯一致）。

### A3. SOAP 类内联（在 baseline `class Muon` 之后插入）

```python
# -----------------------------
# SOAP OPTIMIZER (inlined verbatim from nikhilvyas/modded-nanogpt-SOAP/soap.py)
# -----------------------------
# Reference: https://arxiv.org/abs/2409.11321
# trick: inline SOAP class, no algorithm changes.

class SOAP(optim.Optimizer):
    # ... 来自参考仓库 soap.py 的 class SOAP 全部 410 行（__init__/step/init_preconditioner/
    # project/update_preconditioner/project_back/get_orthogonal_matrix/
    # get_orthogonal_matrix_QR/merge_dims），逐字复制，不改默认值。
```

参考仓库使用 `optim.Optimizer`（baseline 已有 `import torch.optim as optim`）。**不动算法、不动默认值**。

### A4. 优化器实例化（baseline 第 905-912 行）

```python
# trick: SOAP replaces Muon for transformer-block matrix params.
optimizer_soap = SOAP(
    matrix_params,
    lr=args.soap_lr,
    betas=(args.soap_beta1, args.soap_beta2),
    shampoo_beta=args.soap_shampoo_beta,
    eps=args.soap_eps,
    weight_decay=args.soap_weight_decay,
    precondition_frequency=args.soap_precondition_frequency,
    max_precond_dim=args.soap_max_precond_dim,
    merge_dims=args.soap_merge_dims,
    precondition_1d=args.soap_precondition_1d,
    normalize_grads=args.soap_normalize_grads,
    data_format=args.soap_data_format,
    correct_bias=args.soap_correct_bias,
)
for group in optimizer_soap.param_groups:
    group["base_lr"] = args.soap_lr
```

baseline 第 919 行 `optimizers` 列表里 `optimizer_muon` → `optimizer_soap`。其它 3 个优化器（tok / head / scalar）不动。

### A5. 训练 loop 删除 Muon momentum warmup（baseline 第 1111-1114 行）

```python
# trick: SOAP uses fixed betas; Muon momentum warmup (muon_momentum_warmup_*) no longer applied.
```

### A6. 启动日志 + W&B 日志

启动日志（baseline 第 935-939 行附近）追加：
```python
log0(
    f"optimizer:SOAP soap_lr:{args.soap_lr} betas:({args.soap_beta1},{args.soap_beta2}) "
    f"shampoo_beta:{args.soap_shampoo_beta} pf:{args.soap_precondition_frequency} "
    f"max_precond_dim:{args.soap_max_precond_dim} merge_dims:{args.soap_merge_dims}"
)
```

W&B 字典（baseline 第 1143-1147 行）：
- 删除 `"train/muon_momentum"`
- 重命名 `"train/lr_matrix"` → `"train/lr_soap"`，值取自 `optimizer_soap.param_groups[0]["lr"]`

---

## §B `exp/soap/soap.json`

基于 `baseline-sp1024.json`，只改 `trainer_path`（顶层 + 两个 experiment）和 `name`：

```json
{
  "version": 1,
  "trainer_path": "exp/soap/train_gpt.py",
  "launcher": { "nproc_per_node": 8, "master_port_base": 29500 },
  "defaults": { /* 完全照抄 baseline，不增不删 */ },
  "experiments": [
    { "name": "soap-fixed_time_10min", "trainer_path": "exp/soap/train_gpt.py",
      "control": { "mode": "fixed_compute", "target_wallclock_seconds": 600, "iterations_cap": 1000000 } },
    { "name": "soap-fixed_tokens_10b", "trainer_path": "exp/soap/train_gpt.py",
      "control": { "mode": "fixed_tokens", "target_train_tokens": 10000000000 } }
  ]
}
```

---

## §C `exp/soap/README.md`

包含五段：Method overview、Origin（含 commit hash）、Key differences vs. baseline、Impact on training、BPB analysis（占位），以及"如何用 shell env 覆盖"的 Overrides 段。模板见仓库实际文件。

---

## §D `exp/soap/run.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python exp/run_experiments.py exp/soap/soap.json --dry-run
uv run python exp/run_experiments.py exp/soap/soap.json
```

---

## 风险与缓解（要点）

- **eigh NaN**：参考 `soap.py` 自带 `1e-30 * I` 正则与 fp64 fallback；不另加防御。
- **OOM**：本模型最大 dim=1024 ≪ `max_precond_dim=10000`，安全；OOM 时设 `SOAP_MERGE_DIMS=1`。
- **DDP 同步**：DDP 梯度 all-reduce 后位比特一致 → 各 rank 本地 QR 天然同步，无需 broadcast。
- **超参覆盖**：manifest 不转发 `SOAP_*`；用 shell env，例如 `SOAP_LR=1e-3 uv run python exp/run_experiments.py exp/soap/soap.json`。

---

## 验证（仅静态自检，不跑训练）

```bash
# 1) 文件树
ls exp/soap/

# 2) AST 解析：确认 SOAP 与 Muon 类共存
uv run python -c "import ast, pathlib; \
  cls = [n.name for n in ast.parse(pathlib.Path('exp/soap/train_gpt.py').read_text()).body if isinstance(n, ast.ClassDef)]; \
  print(cls); assert 'SOAP' in cls and 'Muon' in cls"

# 3) Manifest dry-run（harness 自带，仅解析配置，不消耗 GPU）
uv run python exp/run_experiments.py exp/soap/soap.json --dry-run

# 4) Git 隔离
git status --short
```

期望：除 `exp/soap/` 与根目录 `soap.md` 外，其它文件无变动；AST 类列表含 `SOAP` 与 `Muon`；dry-run 解析两个 experiment 无报错。

实际训练 / BPB 测量留给后续；当前任务只完成代码修改并静态确认无误。
