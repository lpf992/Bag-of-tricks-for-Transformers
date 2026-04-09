# 三个比赛的训练侧 / 模型架构侧 Trick 汇总

> **GitHub Repo**: <https://github.com/ADIOCLASSMATE/Bag-of-tricks-for-Transformers>

整理范围：
- 比赛：`nanogpt-speedrun`、`parameter-golf`、`slowrun`
- 仅保留：**训练侧（training-side）**、**模型架构侧（architecture-side）**
- 明确排除：**eval 侧**、**量化 / 压缩侧**

---

## 汇总表

| Method | Category | Source | Record | Description |
|---|---|---|---|---|
| Muon family | 优化器 | nanogpt-speedrun, parameter-golf | nanogpt `2024-10-10_Muon`, `2024-10-17_DistributedMuon`, `2025-09-23_MuonCustomSizing`, `2025-09-29_PolarExpress`, `2025-09-30_CustomBatching`; parameter-golf `T05`, `T12`, `T14`, `T15` | 围绕 Muon 本体或其 Newton-Schulz 更新路径做改造，包括分布式化、动量调度、并行更新与参数分组/参数 bank。 |
| SOAP | 优化器 | nanogpt-speedrun | `track_1_short / 2024-10-09_SOAP` | 直接把 baseline optimizer 从 AdamW 切换为 SOAP，是 short track 中独立提取的优化器替换项。 |
| Snoo | 优化器 | nanogpt-speedrun | `track_2_medium / 2025-09-16_Snoo` | 在现有优化器外包一层 look-ahead 式 outer optimizer，用更慢的外层更新平滑训练轨迹。 |
| Muon update smoothing | 优化器 | nanogpt-speedrun | `track_2_medium / 2025-09-17_UpdateSmoothing` | 给 Muon update 增加 EMA smoothing；该 record 还伴随 LR retune，因此这里按主因素保守表述。 |
| Weight decay family | 训练调度 | nanogpt-speedrun, parameter-golf, slowrun | nanogpt `2025-02-08_WeightDecay`, `2025-11-10_CautiousWD`, `2025-12-18_CautiousWDAdam`; parameter-golf `T06`; slowrun Tiny #7 ([42c3912](https://github.com/qlabs-eng/slowrun/blob/42c39127d19bebbb806afd630fa852936da35562/tiny/train.py)), Limited #12 ([20117fb](https://github.com/qlabs-eng/slowrun/blob/20117fb36e62b65e96997268883f0b789a85e75d/train.py)) | 调整 WD 数值、把 cautious WD 扩展到不同参数组，或把 WD 做成显式多阶段 schedule。 |
| Warmdown / cooldown | 训练调度 | nanogpt-speedrun, parameter-golf, slowrun | nanogpt `2025-03-06_LongerCooldown`; parameter-golf `T02`; slowrun Limited #10 ([28152d9](https://github.com/qlabs-eng/slowrun/blob/28152d9996d9e910d4fb1b4a569fae399c546d6b/train.py)) | 改训练后段的学习率衰减长度或比例，本质上是在拉长或重调 late-stage decay。 |
| SWA / EMA | 训练调度 | parameter-golf, slowrun | parameter-golf `T09`, `T10`; slowrun Limited #11 ([3fb4428](https://github.com/qlabs-eng/slowrun/blob/3fb4428f67e77e3bec53d4eafbb2cfd6f999b684/train.py)) | 用 SWA 或 EMA 维护平滑权重；slowrun 文档里明确包含 final-vs-EMA / blend 候选比较。 |
| LowerLR | 训练调度 | parameter-golf | `T01 LowerLR` | 统一下调矩阵、标量和 tied embedding 学习率，是最直接的全局 LR 重调。 |
| Seq / batch scheduling | 训练调度 | nanogpt-speedrun, parameter-golf | nanogpt `2025-01-26_BatchSize`, `2025-11-29_BatchSizeSchedule`; parameter-golf `T03`, `T04` | 通过调整序列长度、batch tokens 或 batch size schedule 重分配训练 token 预算。 |
| GradClip03 | 训练稳定性 | parameter-golf | `T08 GradClip03` | 把梯度裁剪阈值设为 0.3，用更强的梯度范数约束稳定更新。 |
| PerEpochReshuffle | 数据 | slowrun | Limited #2 ([106a290](https://github.com/qlabs-eng/slowrun/blob/106a290604abb6d8c5b0c3cc94c3b0eb6fe87dff/train.py)) | 每个 epoch 重新打乱预分词后的样本顺序，改变训练样本呈现次序。 |
| BosAlign | 数据 | nanogpt-speedrun | `track_1_short / 2025-07-12_BosAlign` | 把训练数据切片起点对齐到 BOS token，减少跨文档切分带来的上下文边界噪声。 |
| DecoderReplayEnable | 数据/训练 | slowrun | Limited #7 ([7d8e580](https://github.com/qlabs-eng/slowrun/blob/7d8e580ab6a339079294562d000df3f7b1ce8c3c/train.py)) | 在训练后段重放 decoder span，让模型重复见到部分片段；不同 track 的具体 replay 细节略有差异。 |
| ReplayScheduleTuning | 训练调度 | slowrun | Limited #8 ([64be473](https://github.com/qlabs-eng/slowrun/blob/64be4733075251c7da1d8b25529963520b16cdb8/train.py)) | 进一步调 replay 的开始时机、循环次数和 span，属于 replay 机制上的调度细化。 |
| U-Net / skip family | 结构 | nanogpt-speedrun, slowrun | nanogpt `2025-11-18_RefineSkip`; slowrun Limited #5 ([e463653](https://github.com/qlabs-eng/slowrun/blob/e463653a2b07790e0694bfaa6bdd7e6ee57cef64/train.py)) | 在层间加入 encoder-decoder 式镜像 skip；nanogpt 版本进一步把 skip 结构从大 U 收紧到更精简的连接。 |
| Value path family | 结构 | nanogpt-speedrun, parameter-golf, slowrun | nanogpt `2024-12-04_ValueEmbed`, `2025-04-22_Record8`, `2025-08-28_Medium_NewValemb`, `2025-10-04_GPT2MediumLayerReuse`, `2026-01-26-UntieValueEmbeddings`; parameter-golf `A10`; slowrun Limited #3 ([b261fba](https://github.com/qlabs-eng/slowrun/blob/b261fba252920582076cf8c77dedf9251fe7f7ed/train.py)) | 给网络增加额外的 value 注入路径：包括多层 VE、x0 投影、layer reuse，以及 tied/untied 的 value embedding 变体。 |
| UntieEmbed | 结构 | nanogpt-speedrun | `track_1_short / 2024-11-03_UntieEmbed` | 主因素是把 embedding 与 lm head 从 tied 改成 untied，并配合少量头部归一化/初始化调整。 |
| BigramHash | 结构 | nanogpt-speedrun, parameter-golf | nanogpt `2026-01-19_BigramHashEmbedding`; parameter-golf `A05` | 额外引入 hash-based bigram embedding 分支，用哈希桶表示二元 token 组合后再并入主干。 |
| Smear | 结构 | nanogpt-speedrun | `track_1_short / 2025-09-18_Smear` | 把 token embedding 向前 smear 一个位置，属于显式的相邻 token 混合。 |
| SmearGate | 结构 | parameter-golf | `A04 SmearGate` | 用可学习 gate 混合相邻 token embedding；和简单 smear 相近，但机制是 gated blend。 |
| SwiGLU | 结构 | slowrun | Limited #4 ([22d4a24](https://github.com/qlabs-eng/slowrun/blob/22d4a24ec53633c16d643779900ac3e9d10643a3/train.py)) | 把 ReLU^2 MLP 换成带门控的 SwiGLU MLP，用 gated product 重写 FFN 路径。 |
| LeakyReLUSquared | 结构 | parameter-golf | `T13 LeakyReLUSquared` | 把 MLP 激活写成 `leaky_relu(x, 0.5).square()`，是对非线性形状的轻量替换。 |
| MLPx3 | 结构 | parameter-golf | `A03 MLPx3` | 把 MLP 扩到 3 倍宽隐藏层，在相同骨架下提高 FFN 容量。 |
| 10Layers / 11Layers | 结构 | parameter-golf | `A01 10Layers`, `A02 11Layers` | 直接增加 Transformer 层数，是最直接的深度扩展尝试。 |
| LNScale | 结构/数值 | parameter-golf | `A09 LNScale` | 在 RMSNorm 输出后再乘上随层数缩放的系数 `1/sqrt(l+1)`，抑制深层幅值增长。 |
| PartialRoPE | 结构 | parameter-golf | `A08 PartialRoPE` | 只对部分 head 维度施加 RoPE，把其余维度保留为非旋转通道。 |
| HalfTruncatedRoPE | 结构 | slowrun | Tiny #6 ([ed62160](https://github.com/qlabs-eng/slowrun/blob/ed62160275273197c3a996c4469d735a05c5eedb/tiny/train.py)) | 只旋转一半的 head 维度，另一半保持静止；在 slowrun 中与 key-offset 改动同属一组 bundle。 |
| PartialKeyOffset | 结构 | nanogpt-speedrun, slowrun | nanogpt `2025-12-14_PartialKeyOffset`; slowrun Tiny #6 ([ed62160](https://github.com/qlabs-eng/slowrun/blob/ed62160275273197c3a996c4469d735a05c5eedb/tiny/train.py)) | 对长窗口层的部分 key 维度做前移/offset，改变局部注意力中的 key 对齐方式。 |
| AttentionHeadGating | 结构 | slowrun | Limited #6 ([52e7441](https://github.com/qlabs-eng/slowrun/blob/52e7441f862c3295c0f5695933438dac78f7fc5b/train.py)) | 在 attention 输出端加入 per-head sigmoid gate，让每个头的贡献可学习缩放。 |
| SparseAttnGate | 结构 | nanogpt-speedrun | `track_1_short / 2025-08-23_SparseAttnGate` | 给 attention 路径加 sparse gate，主目标是让部分注意力通道更稀疏地参与计算。 |
| DropAttn | 结构 | nanogpt-speedrun | `track_1_short / 2025-09-21_DropAttn` | 主因素是去掉第一层 attention，在浅层直接减掉一段注意力计算路径。 |
| PairedHeadAttention | 结构 | nanogpt-speedrun | `track_1_short / 2026-01-07_PairedHeadAttention` | 以成对 head 的方式重组注意力，是一次 head-level 的结构重排尝试。 |
| XSA family | 结构 | parameter-golf, slowrun | parameter-golf `A06`, `A07`; slowrun Limited #9 ([ac968a6](https://github.com/qlabs-eng/slowrun/blob/ac968a62c633d75d972afa6d86a59f89e12997b9/train.py)) | XSA 会移除 attention 输出中的 self-value 分量；parameter-golf 进一步区分只加在后 4 层或全层。 |
| YaRN | 长上下文 | nanogpt-speedrun, parameter-golf | nanogpt `2025-09-10_Yarn`; parameter-golf `A14` | 对 RoPE 频率做长上下文外推/缩放，让模型在更长上下文下维持位置编码可用性。 |
| OrthoInit | 初始化 | parameter-golf | `T07 OrthoInit` | 用 `nn.init.orthogonal_` 配合 muP 输出缩放，调整线性层初始谱性质与输出尺度。 |
| ResidualLambdaInit11 | 初始化 | slowrun | Tiny #6 ([ed62160](https://github.com/qlabs-eng/slowrun/blob/ed62160275273197c3a996c4469d735a05c5eedb/tiny/train.py)) | 把 residual lambda 初值从 1.0 调到 1.1，是对残差分支初始强度的小幅改动。 |
| SALambdaOnWeights | 结构/数值 | nanogpt-speedrun | `track_1_short / 2025-12-10_SALambdaOnWeights` | 把 self-attention lambda 的乘法位置前移到 QKV/权重侧，而不是更后面的激活侧。 |
| SoftCap / LogitRescale / PolynomialSoftcap | 数值 | nanogpt-speedrun, parameter-golf | nanogpt `2025-01-04_SoftCap`, `2025-12-26_LogitRescale`; parameter-golf `A13` | 通过 softcap 或多项式 clamping/rescale 约束 logits 幅值，减少极端 logit 带来的数值放大。 |
| MultiTokenPrediction | 训练目标 | nanogpt-speedrun | `track_1_short / 2025-12-22_MultiTokenPrediction` | 把单步 next-token 预测扩成 multi-token prediction，用一个前向同时监督多个未来位置。 |
| AdamSyncGradientHook | 工程 | nanogpt-speedrun | `track_1_short / 2025-10-31_AdamSyncGradientHook` | 把 DistAdam 的梯度同步前移到 backward hooks，减少显式同步阶段的组织开销。 |
| 8192BPE | 词表/Tokenizer | parameter-golf | `A11 8192BPE` | 换用 8192-token 的独立 BPE 词表，是 tokenizer 层面的容量/压缩重分配。 |
| FactoredEmbedding | 结构 | parameter-golf | `A12 FactoredEmbedding` | 用低维 bottleneck 投影实现 factorized embedding，在词表侧节省参数后再回投到模型维度。 |
