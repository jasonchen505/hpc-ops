# LLM 算法实习面试准备指南

> 基于 HPC-Ops 项目深度分析，聚焦 LLM & Agent 应用及后训练相关知识点

---

## 目录

1. [项目概述与学习路径](#1-项目概述与学习路径)
2. [Attention 机制深度解析](#2-attention-机制深度解析)
3. [MoE (Mixture of Experts) 架构](#3-moe-mixture-of-experts-架构)
4. [量化技术与精度优化](#4-量化技术与精度优化)
5. [推理优化核心技巧](#5-推理优化核心技巧)
6. [分布式推理与并行策略](#6-分布式推理与并行策略)
7. [采样与解码策略](#7-采样与解码策略)
8. [高频面试问题与深度挖掘点](#8-高频面试问题与深度挖掘点)
9. [项目经验介绍模板](#9-项目经验介绍模板)
10. [延伸学习资源](#10-延伸学习资源)

---

## 1. 项目概述与学习路径

### 1.1 HPC-Ops 项目定位

HPC-Ops 是腾讯混元 AI Infra 团队开发的**生产级高性能 LLM 推理算子库**，专注于：

- **Attention** (Prefill/Decode)
- **MoE** (Mixture of Experts)
- **GEMM** (矩阵乘法)
- **Sampling** (采样)
- **Normalization** (归一化)
- **Communication-Compute Fusion** (通信计算融合)

**目标硬件**: NVIDIA H20 GPU (SM90 架构)

### 1.2 学习路径建议

```
Week 1: 理解 LLM 推理流程 + Attention 机制
Week 2: 深入 KV Cache、Paged Attention、Prefill/Decode 差异
Week 3: 学习 MoE 架构原理与实现
Week 4: 掌握量化技术 (FP8, per-tensor/per-token)
Week 5: 理解分布式推理 (TP/EP) + 通信优化
Week 6: 复习采样策略 + 准备项目介绍
```

---

## 2. Attention 机制深度解析

### 2.1 核心公式与计算流程

标准 Attention:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**面试考察点**: 为什么要除以 √d_k？
- 防止点积过大导致 softmax 梯度消失
- d_k 是 head dimension，通常为 128

### 2.2 KV Cache 机制

**为什么需要 KV Cache？**
- 自回归生成中，每个 token 只需要计算自己的 Q，但需要与所有历史 token 的 K、V 做 attention
- 缓存历史 K、V 避免重复计算

**代码位置**: `hpc/attention.py:70-145`

```python
def attention_with_kvcache_prefill_bf16(
    q: Tensor,           # [total_seq, num_head_q, num_dim_qk]
    kcache: Tensor,      # [num_blocks, block_size, num_head_kv, num_dim_qk]
    vcache: Tensor,      # [num_blocks, block_size, num_head_kv, num_dim_v]
    cu_seqlens_q: Tensor,
    block_ids: Tensor,   # [num_batch, max_blocks]
    seqlens_kvcache: Tensor,
    ...
)
```

**面试深度挖掘**:

1. **Paged KV Cache 的设计动机是什么？**
   - 传统连续内存分配导致内存碎片化
   - Paged Attention 将 KV Cache 分成固定大小的 block (通常 16 tokens)
   - 通过 page table (block_ids) 管理，支持动态分配和释放

2. **NHD vs HND 布局有什么区别？**
   - NHD: `[num_blocks, block_size, num_head_kv, dim]` - batch 优先
   - HND: `[num_head_kv, num_blocks, block_size, dim]` - head 优先
   - HND 在多 head 并行时更友好，但需要 stride 变换

3. **Prefill 和 Decode 阶段的 Attention 有什么本质区别？**
   - **Prefill**: 处理整个 prompt，Q 有多个 token，计算密集型
   - **Decode**: 每次只生成 1 个 token，Q 只有 1 个 token，访存密集型
   - 优化策略完全不同：Prefill 用大 tile，Decode 用 split-K

### 2.3 FP8 Attention 与量化

**代码位置**: `hpc/attention.py:148-250`

```python
class QuantType(Enum):
    QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD = 0
    QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR = 1
    QPERTENSOR_KPERTENSOR_VPERTENSOR = 2
```

**面试考察点**:

1. **FP8 量化为什么重要？**
   - 内存减半：FP8 (1 byte) vs BF16 (2 bytes)
   - 计算加速：H20 GPU 的 FP8 Tensor Core 吞吐是 BF16 的 2 倍
   - KV Cache 容量翻倍，支持更长上下文

2. **Per-tensor vs Per-token vs Per-head 量化有什么区别？**
   - **Per-tensor**: 整个 tensor 一个 scale，最简单但精度损失大
   - **Per-token**: 每个 token 一个 scale，精度更好
   - **Per-head**: 每个 head 一个 scale，精度最好但开销大
   - HPC-Ops 默认用 Q per-token per-head + K/V per-tensor (quant_type=1)

3. **量化误差如何累积？如何缓解？**
   - Q@K 的误差 = Q 量化误差 + K 量化误差
   - 使用 per-token 量化可以自适应不同 token 的动态范围
   - 代码中 `qscale` shape 为 `[num_batch, num_head_q, max_seqlens_q_pad]`

### 2.4 Block-Sparse Attention

**代码位置**: `hpc/attention.py:253-338`

```python
def attention_with_kvcache_blocksparse_prefill_fp8(
    ...
    block_mask: Optional[Tensor] = None,  # [num_batch, num_head_q, max_tile_m, num_tile_kv_in_mask]
    ...
)
```

**面试考察点**:

1. **为什么需要 Sparse Attention？**
   - 长上下文 (128K+) 时，大部分 KV token 与当前 query 无关
   - 稀疏 attention 跳过不相关的 KV tile，减少计算量
   - 实测最高 3.16x 加速

2. **Block-level sparsity vs Token-level sparsity？**
   - Block-level: 以 tile 为单位跳过，对硬件更友好
   - Token-level: 更精细但管理开销大
   - HPC-Ops 用 block-level，block size 通常 64 或 128

3. **如何生成 block mask？**
   - 可以用预计算的 attention pattern
   - 或者用 STEM 算子动态生成 (`hpc/stem.py`)

### 2.5 动态 Decode Attention 调度

**代码位置**: `hpc/attention.py:515-692`

**核心思想**: 静态 split-K 无法适应动态长度分布，导致长请求尾部延迟高、短请求资源浪费。

```python
# 1. 分配 workspace
task_map = get_attention_decode_task_workspace(max_num_batch, max_seqlen, num_head_kv)

# 2. 分配任务 (greedy bin-packing)
assign_attention_decode_task(num_seq_kvcache, task_map, num_head_kv, ...)

# 3. 使用 task_map 执行 attention
output = attention_decode_fp8(q, kcache, vcache, ..., task_map=task_map)
```

**面试深度挖掘**:

1. **Greedy bin-packing 策略是什么？**
   - 将所有请求切成统一大小的 KV tile
   - 贪心地将 tile 分配到负载最小的 CTA
   - 类似于装箱问题的在线算法

2. **为什么静态 split-K 效率低？**
   - 假设 split-K=4，但某个请求 KV 长度 10000，其他请求只有 100
   - 长请求的 4 个 split 仍然很长，短请求的 4 个 split 很短
   - CTA 之间负载不均衡

3. **动态调度的开销如何控制？**
   - Task map 预分配，避免每次分配内存
   - CPU 端预计算，不占用 GPU 计算资源
   - Task map 结构紧凑 (每个 task 48 bytes)

---

## 3. MoE (Mixture of Experts) 架构

### 3.1 MoE 基本原理

**核心思想**: 用多个 expert 处理不同类型的 token，通过 router 选择 top-k experts。

```
output = Σ(topk_scale[i] * Expert[i](input)) + shared_output
```

**代码位置**: `hpc/fuse_moe.py`

### 3.2 Fused MoE 流程

**关键代码** (`src/fuse_moe/fuse_moe.cu:14-60`):

```cpp
// 0. Token 排序: 按 expert 分组
count_and_gather_async(...)

// 1. Gate-Up GEMM: [hidden_size] -> [intermediate_size * 2]
group_gemm_fp8_async(gate_up_output, gate_up_input, gate_up_weight, ...)

// 2. 激活 + 量化: SiLU(gate) * up -> FP8
act_mul_and_quant_async(down_input, gate_up_output, ...)

// 3. Down GEMM: [intermediate_size] -> [hidden_size]
group_gemm_fp8_async(down_output, down_input, down_weight, ...)

// 4. 加权归约: scatter-add
reduce_async(output, down_output, topk_pos, topk_scale, ...)
```

**面试考察点**:

1. **为什么需要 Fused MoE？**
   - 传统实现: gather -> GEMM -> scatter，多次内存读写
   - Fused 实现: 融合所有步骤，减少中间数据落地
   - 实测最高 1.6x 加速 (TP=8)

2. **Gate-Up GEMM 和 Down GEMM 的区别？**
   - Gate-Up: `[num_tokens, hidden_size] x [num_experts, intermediate_size*2, hidden_size]`
   - Down: `[num_tokens, intermediate_size] x [num_experts, hidden_size, intermediate_size]`
   - 中间有 SiLU 激活和 FP8 量化

3. **为什么用 SiLU 激活而不是 ReLU？**
   - SiLU(x) = x * sigmoid(x)，平滑且非单调
   - 在 LLM 中效果更好 (GPT, LLaMA 都用 SiLU)
   - 代码中 `silu(gate) * up` 是标准的 SwiGLU 变体

### 3.3 Expert Parallelism (EP)

**代码位置**: `hpc/fuse_moe.py:8-85`

```python
def count_and_gather(
    x: Tensor,
    topk_ids: Tensor,
    num_expert: int,      # 当前设备的 expert 数量
    rank_ep: int,         # 当前设备在 EP 中的 rank
    ...
)
```

**面试考察点**:

1. **TP vs EP 的区别？**
   - **Tensor Parallelism (TP)**: 将每个 expert 的权重切分到多个 GPU
   - **Expert Parallelism (EP)**: 将不同 expert 分配到不同 GPU
   - TP=8, EP=1: 每个 GPU 都有所有 expert 的一部分
   - TP=1, EP=8: 每个 GPU 只有部分 expert

2. **EP 的通信开销在哪里？**
   - All-to-All: 需要将 token 发送到对应的 expert 所在 GPU
   - Reduce-Scatter: 收集所有 expert 的输出并加权求和
   - HPC-Ops 通过优化 count_and_gather 减少通信

3. **DeepSeek-V3 的 MoE 设计有什么特点？**
   - 256 experts, top-8 routing
   - 每个 expert 的 intermediate_size 较小 (2048)
   - 适合 EP=8，每个 GPU 32 experts

### 3.4 TMA vs cp.async 路径选择

**代码位置**: `src/fuse_moe/fuse_moe.cu` vs `src/fuse_moe/cp_async/fuse_moe.cu`

**面试考察点**:

1. **什么是 TMA (Tensor Memory Accelerator)？**
   - SM90 (H20) 的新硬件特性
   - 可以异步加载多维 tensor，支持越界检测
   - 减少 warp 的内存访问负担

2. **什么时候用 cp.async 而不用 TMA？**
   - 当 intermediate_size <= 512 时，用 cp.async
   - 小尺寸时 TMA 的配置开销反而更大
   - cp.async 更简单，CTA 占用率更高

3. **Warp Specialization 是什么？**
   - 将 warp 分为 producer (数据加载) 和 consumer (计算)
   - TMA 路径用 warp specialization 实现 pipeline
   - cp.async 路径不用，因为小尺寸时 pipeline 收益不大

---

## 4. 量化技术与精度优化

### 4.1 FP8 量化基础

**FP8 格式**:
- **E4M3**: 4 位指数 + 3 位尾数，范围 [-448, 448]，精度较高
- **E5M2**: 5 位指数 + 2 位尾数，范围 [-57344, 57344]，范围较大
- HPC-Ops 使用 E4M3 (float8_e4m3)

**代码位置**: `hpc/act.py`

```python
def scaled_fp8_quant(
    x: Tensor,           # BF16 input
    scale: Tensor,       # FP8 scale
) -> Tensor:             # FP8 output
```

### 4.2 量化策略对比

| 策略 | Scale Shape | 精度 | 开销 | 适用场景 |
|------|-------------|------|------|----------|
| Per-tensor | `[1]` | 低 | 最小 | 对精度不敏感的场景 |
| Per-token | `[num_tokens]` | 中 | 小 | 通用场景 |
| Per-head | `[num_heads]` | 高 | 中 | 精度敏感场景 |
| Per-token-per-head | `[num_tokens, num_heads]` | 最高 | 大 | 最高精度需求 |

**面试深度挖掘**:

1. **为什么 K/V 通常用 per-tensor 而 Q 用 per-token？**
   - K/V 存在 KV Cache 中，per-token scale 需要额外存储
   - Q 每次重新计算，per-token scale 开销可接受
   - 实际中 K/V 的动态范围比 Q 更稳定

2. **FP8 量化如何影响 Attention Score？**
   ```
   Attention = softmax(Q_fp8 * K_fp8^T * qscale * kscale / √d_k) * V_fp8 * vscale
   ```
   - 量化误差在 softmax 前累积
   - 但 softmax 会放大差异，所以 Q/K 的精度更重要

3. **Block-wise quantization 是什么？**
   - 将 tensor 分成固定大小的 block (如 128 元素)
   - 每个 block 一个 scale
   - 比 per-tensor 精度好，比 per-token 开销小

### 4.3 BF16 x FP32 混合精度 GEMM

**代码位置**: `hpc/gemm.py`

```python
def gemm_bf16xfp32(
    x: Tensor,      # BF16
    w: Tensor,      # FP32
) -> Tensor:        # BF16
```

**核心思想**: FP32 权重分解为两个 BF16 分量
```python
scale = 1 / 256
w_high = w_fp32.to(torch.bfloat16)
w_low = ((w_fp32 - w_high.float()) / scale).to(torch.bfloat16)
# Result = x @ w_high + scale * (x @ w_low)
```

**面试考察点**:

1. **为什么需要 BF16 x FP32 GEMM？**
   - MoE router 对精度敏感，权重需要 FP32
   - 但 FP32 Tensor Core 吞吐低 (H20 没有原生 FP32 TC)
   - 分解成两个 BF16 GEMM，保持精度同时提升吞吐

2. **为什么 scale 用 1/256？**
   - 256 = 2^8，可以用位移操作，开销小
   - 保证 w_low 在 BF16 范围内
   - 实测精度损失可忽略

3. **这个技巧可以用在其他地方吗？**
   - 状态压缩 GEMM (sparse/linear attention)
   - 任何需要 FP32 精度但可以容忍小误差的场景

---

## 5. 推理优化核心技巧

### 5.1 Kernel Fusion (算子融合)

**核心思想**: 将多个小 kernel 合并成一个大 kernel，减少：
- Kernel launch 开销
- 中间数据的 HBM 读写
- 同步等待

**代码示例**: Fused AllReduce + RMSNorm

```python
def fused_allreduce_rmsnorm(
    x: Tensor,           # 输入
    residual: Tensor,    # 残差
    weight: Tensor,      # RMSNorm 权重
    ...
) -> Tensor:
    # 融合了: AllReduce + Residual Add + RMSNorm
```

**面试考察点**:

1. **AllReduce + RMSNorm 为什么要融合？**
   - 传统流程: AllReduce -> 写回 HBM -> 读取 -> Residual Add -> RMSNorm -> 写回
   - 融合后: AllReduce 过程中直接计算，只写回最终结果
   - 减少 2 次 HBM 读写

2. **High-throughput vs Low-latency 模式？**
   - **High-throughput**: 用 CUDA multicast，适合大 batch (prefill)
   - **Low-latency**: 用 Lamport P2P，适合小 batch (decode)
   - 自动选择：根据 num_tokens 决定

3. **什么是 Lamport 通信？**
   - 无锁通信协议，通过标志位检测数据就绪
   - 避免传统的 barrier 同步开销
   - 代码中 `LamportFlags` 管理多个 stage

### 5.2 Task Scheduling (任务调度)

**核心思想**: 预计算任务分配，避免运行时调度开销。

**代码位置**: `hpc/attention.py:515-621`

**面试考察点**:

1. **为什么不在 kernel 内部做动态调度？**
   - GPU kernel 内部做调度逻辑复杂且开销大
   - CPU 端预计算可以利用更复杂的算法
   - Task map 结构简单，传递开销小

2. **Greedy bin-packing 的时间复杂度？**
   - O(n log n)，其中 n 是任务数
   - 可以在 CPU 端并行化
   - 实测对 decode 延迟影响 <1%

3. **如何处理动态长度变化？**
   - 每个 decode step 重新调用 assign_attention_decode_task
   - Task map 预分配，避免内存分配
   - 支持 MTP (Multi-Token Prediction)

### 5.3 PDL (Programmatic Dependent Launch)

**核心思想**: 允许 kernel 在前一个 kernel 完成前启动，减少 launch bubble。

**代码位置**: `src/fuse_moe/fuse_moe.cu:30`

```cpp
bool use_pdl = true;
```

**面试考察点**:

1. **PDL 和 CUDA Stream 的区别？**
   - CUDA Stream: 不同 stream 的 kernel 可以并行
   - PDL: 同一 stream 的 kernel 可以 overlap
   - PDL 需要 `cudaGridDependencySynchronize()` 和 `cudaTriggerProgrammaticLaunchCompletion()`

2. **PDL 在 Fused MoE 中如何工作？**
   - Gate-Up GEMM 和 activation 可以 overlap
   - Activation 和 Down GEMM 可以 overlap
   - 减少 kernel launch bubble

### 5.4 TMA (Tensor Memory Accelerator)

**核心思想**: SM90 的硬件内存访问单元，支持多维 tensor 的异步加载。

**代码位置**: `src/utils/tma.cuh`

**面试考察点**:

1. **TMA 相比 cp.async 的优势？**
   - 支持多维 tensor，不需要手动计算 stride
   - 支持越界检测，可以安全加载不规则形状
   - 减少 warp 的内存访问负担

2. **TMA Descriptor 是什么？**
   - 描述 tensor 的形状、stride、数据类型
   - 预先配置，kernel 运行时直接使用
   - 代码中 `tmas` tensor 存储 TMA descriptor

---

## 6. 分布式推理与并行策略

### 6.1 Tensor Parallelism (TP)

**核心思想**: 将模型的权重矩阵按行或列切分到多个 GPU。

**代码位置**: `hpc/allreduce.py`

```python
def fused_allreduce_rmsnorm(
    x: Tensor,           # 当前 TP rank 的输出
    residual: Tensor,
    weight: Tensor,
    ...
) -> Tensor:
    # AllReduce: 汇总所有 TP rank 的结果
    # RMSNorm: 归一化
```

**面试考察点**:

1. **Column Parallel vs Row Parallel？**
   - **Column Parallel**: 输出按列切分，需要 AllGather 收集
   - **Row Parallel**: 输入按行切分，需要 ReduceScatter 汇总
   - 通常 Layer 1 用 Column，Layer 2 用 Row，减少通信

2. **TP 的通信瓶颈在哪里？**
   - AllReduce 需要两次通信: ReduceScatter + AllGather
   - 通信量 = 2 * hidden_size * num_tokens
   - HPC-Ops 融合 RMSNorm 减少一次 HBM 读写

3. **NVLink vs NVSwitch 的区别？**
   - NVLink: GPU 之间的直接连接，带宽 900 GB/s (H20)
   - NVSwitch: 连接所有 GPU 的交换芯片
   - HPC-Ops 使用 multicast 利用 NVSwitch

### 6.2 Expert Parallelism (EP)

**核心思想**: 将不同的 expert 分配到不同的 GPU。

**代码位置**: `hpc/fuse_moe.py:8-85`

**面试考察点**:

1. **EP 的 All-to-All 通信模式？**
   - 每个 GPU 需要将 token 发送到对应的 expert 所在 GPU
   - 通信模式不规则，难以优化
   - HPC-Ops 通过 count_and_gather 减少通信量

2. **TP + EP 混合并行？**
   - DeepSeek-V3: TP=8, EP=1 或 TP=1, EP=8
   - TP=8: 每个 GPU 有所有 expert 的一部分权重
   - EP=8: 每个 GPU 有部分 expert 的完整权重

3. **EP 的负载均衡问题？**
   - 某些 expert 可能被更多 token 选择
   - 需要动态调整 expert 分配
   - HPC-Ops 通过 task scheduling 缓解

### 6.3 Multicast 通信

**核心思想**: 利用 NVLink 的 multicast 能力，一次发送给多个 GPU。

**代码位置**: `src/allreduce/fuse_allreduce_rmsnorm_high_throughput.cu`

**面试考察点**:

1. **什么是 CUDA Multicast？**
   - 一条指令可以同时写入多个 GPU 的内存
   - 减少通信次数，提高带宽利用率
   - 需要 NVSwitch 支持

2. **Multicast vs Unicast 的性能差异？**
   - Multicast: 1 次发送 -> N 个 GPU 收到
   - Unicast: N 次发送 -> N 个 GPU 收到
   - 理论上 N 倍加速，但实际受限于 NVLink 带宽

---

## 7. 采样与解码策略

### 7.1 Fused Sampler

**代码位置**: `hpc/sampler.py`

```python
def fused_sampler(
    logits: Tensor,
    *,
    penalty_mask: Optional[Tensor] = None,
    repetition_penalty: Union[Tensor, float] = 0.0,
    temperature: Union[Tensor, float] = 0.0,
    softmax_policy: SoftmaxPolicy = SoftmaxPolicy.NONE,
    topk: Union[Tensor, int] = 0,
    topp: Union[Tensor, float] = 0.0,
    ...
) -> Tensor:
```

**流水线**:
```
repetition_penalty -> temperature -> [softmax1] ->
topk -> [softmax2] -> topp -> Gumbel-max sampling -> penalty writeback
```

**面试考察点**:

1. **Top-K 和 Top-P 采样的区别？**
   - **Top-K**: 固定选择 K 个最可能的 token
   - **Top-P (Nucleus)**: 选择累积概率超过 P 的最小 token 集合
   - Top-P 更自适应，但计算更复杂

2. **为什么需要 Fused Sampler？**
   - 小 batch 时，多个小 kernel 的 launch 开销占比高
   - 融合成 2 个 kernel，减少 launch 开销
   - 实测最高 8.5x 加速

3. **Temperature 的作用？**
   - T < 1: 分布更尖锐，输出更确定
   - T > 1: 分布更平滑，输出更随机
   - T = 0: 等价于 argmax (greedy)

4. **Repetition Penalty 如何实现？**
   - 维护 penalty_mask 记录已生成的 token
   - 对已生成的 token 的 logit 乘以惩罚系数
   - 代码中用 bitmask 存储，节省内存

### 7.2 Speculative Decoding

**相关概念**: MTP (Multi-Token Prediction)

**代码位置**: `hpc/attention.py:341-412`

```python
def attention_decode_bf16(
    ...
    mtp: int = 0,              # number draft tokens
    ...
)
```

**面试考察点**:

1. **什么是 Speculative Decoding？**
   - 用小模型 (draft model) 快速生成多个候选 token
   - 用大模型 (target model) 并行验证
   - 接受正确的 token，拒绝错误的 token

2. **MTP 如何加速？**
   - 一次生成多个 token，减少 decode step
   - 需要并行验证多个位置的 attention
   - HPC-Ops 的 attention kernel 支持 mtp > 0

3. **Acceptance Rate 如何提高？**
   - 使用 Medusa 风格的多头预测
   - 或者用 n-gram 模式匹配
   - 关键是 draft model 要足够快且准确

---

## 8. 高频面试问题与深度挖掘点

### 8.1 Attention 相关

**Q1: 为什么 H20 GPU 的 decode attention 比 prefill 更难优化？**

深度挖掘点:
- Decode 阶段每个请求只有 1 个 query token，但可能有很长的 KV cache
- 计算量小但访存量大，是 memory-bound 场景
- 需要用 split-K 技术将 KV 切分到多个 CTA 并行处理
- 动态长度分布导致负载不均衡

**Q2: Paged Attention 的内存碎片问题如何解决？**

深度挖掘点:
- 传统连续分配: 预分配最大长度，浪费内存
- Paged Attention: 按需分配 block，类似操作系统虚拟内存
- 但需要管理 free list，有分配开销
- HPC-Ops 用 block_ids 管理 page table

**Q3: FP8 Attention 的精度损失如何评估？**

深度挖掘点:
- 对比 BF16 和 FP8 的 attention output
- 用 MAE (Mean Absolute Error) 和 MRE (Mean Relative Error)
- 关注 top-K 位置的一致性
- HPC-Ops 的测试框架 (`tests/utils.py`) 提供了完善的误差分析

### 8.2 MoE 相关

**Q4: MoE 的 load balancing 问题如何解决？**

深度挖掘点:
- **Expert-level**: 某些 expert 被更多 token 选择
- **Token-level**: 某些 token 无法分配到合适的 expert
- 解决方案:
  - Auxiliary loss 鼓励均匀分配
  - Expert capacity factor 限制最大 token 数
  - Token dropping 丢弃超载的 token

**Q5: Fused MoE 中的 activation 为什么用 SiLU 而不是 GeLU？**

深度挖掘点:
- SiLU(x) = x * sigmoid(x)，计算更快
- GeLU(x) = x * Φ(x)，需要 erf 函数
- 在 LLM 中效果相当，但 SiLU 更容易融合
- SwiGLU (SiLU + Gate) 是目前主流选择

**Q6: EP 模式下如何减少 All-to-All 通信？**

深度挖掘点:
- **Token dropping**: 丢弃需要跨节点通信的 token
- **Expert replication**: 在多个 GPU 上复制热门 expert
- **Hierarchical EP**: 先节点内 EP，再节点间 EP
- HPC-Ops 通过 count_and_gather 优化通信模式

### 8.3 量化相关

**Q7: 量化对模型质量的影响如何评估？**

深度挖掘点:
- **Perplexity**: 在验证集上测量困惑度变化
- **Downstream tasks**: 在 benchmark 上测量准确率
- **Human evaluation**: 人工评估生成质量
- 通常 FP8 量化质量损失 <1%

**Q8: 为什么 KV Cache 量化比 Weight 量化更敏感？**

深度挖掘点:
- KV Cache 的动态范围比权重大
- 不同 token 的 KV 分布差异大
- 量化误差会在 attention 计算中累积
- 所以 Q 用 per-token 量化，K/V 用 per-tensor 量化

**Q9: Mixed-precision 推理的策略是什么？**

深度挖掘点:
- **Attention**: FP8 for KV Cache, BF16 for Q
- **FFN**: FP8 for weights, BF16 for activations
- **Router**: BF16 or FP32 (精度敏感)
- **Sampler**: FP32 for logits

### 8.4 优化相关

**Q10: Kernel Fusion 的收益边界在哪里？**

深度挖掘点:
- **Fusion too small**: launch 开销减少有限
- **Fusion too large**: register spill 增加，occupancy 下降
- **Optimal**: 平衡 launch 开销和 register 使用
- HPC-Ops 的 Fused Sampler 融合成 2 个 kernel，而不是 1 个

**Q11: 如何选择 TMA vs cp.async？**

深度挖掘点:
- **TMA**: 大 tensor，规则形状，SM90+
- **cp.async**: 小 tensor，不规则形状，SM80+
- HPC-Ops 根据 intermediate_size 阈值 (512) 自动选择
- TMA 有配置开销，小 tensor 时 cp.async 更快

**Q12: PDL 的适用场景是什么？**

深度挖掘点:
- **适用**: 多个 kernel 有数据依赖，但计算可以 overlap
- **不适用**: 单个 kernel，或 kernel 间无依赖
- HPC-Ops 在 Fused MoE 中使用 PDL，让 Gate-Up GEMM 和 activation overlap
- 需要显式管理依赖: `cudaGridDependencySynchronize()` 和 `cudaTriggerProgrammaticLaunchCompletion()`

---

## 9. 项目经验介绍模板

### 9.1 一句话介绍

> 我参与了腾讯混元 HPC-Ops 项目，这是一个面向 LLM 推理的高性能算子库，我在其中学习了 Attention、MoE、量化、分布式推理等核心优化技术，并深入理解了 CUDA kernel 设计和 GPU 架构。

### 9.2 详细介绍 (2-3 分钟)

**背景**:
- HPC-Ops 是腾讯混元团队开发的生产级推理算子库
- 目标是在 H20 GPU 上实现 SOTA 性能
- 覆盖 Attention、MoE、GEMM、Sampler、AllReduce 等核心算子

**我的学习收获**:

1. **Attention 优化**:
   - 理解了 KV Cache 和 Paged Attention 的设计原理
   - 学习了 FP8 量化的 per-tensor 和 per-token 策略
   - 掌握了 split-K 和动态任务调度优化 decode attention

2. **MoE 架构**:
   - 理解了 MoE 的路由机制和专家选择
   - 学习了 Fused MoE 的 kernel 设计，包括 TMA 和 cp.async 两条路径
   - 掌握了 EP 和 TP 的通信模式

3. **推理优化技巧**:
   - Kernel Fusion: AllReduce + RMSNorm 融合
   - Task Scheduling: Greedy bin-packing 动态调度
   - PDL: 程序化依赖启动减少 launch bubble

4. **工程实践**:
   - 学习了 PyTorch 算子注册和 fake tensor 机制
   - 掌握了 CUDA kernel 的调试和性能分析方法
   - 理解了生产级代码的质量要求

### 9.3 可以深入讨论的点

根据面试官的兴趣，可以选择深入：

- **如果面试官关注 Attention**: 讨论 KV Cache 设计、FP8 量化策略、动态调度
- **如果面试官关注 MoE**: 讨论 Fused MoE 流程、EP/TP 选择、负载均衡
- **如果面试官关注优化**: 讨论 Kernel Fusion、PDL、TMA vs cp.async
- **如果面试官关注分布式**: 讨论 AllReduce、Multicast、Lamport 通信

---

## 10. 延伸学习资源

### 10.1 论文推荐

1. **Attention**:
   - FlashAttention (Tri Dao)
   - FlashAttention-2 (Tri Dao)
   - FlashAttention-3 (Tri Dao)

2. **MoE**:
   - Switch Transformer (Google)
   - DeepSeek-V3 (DeepSeek)
   - Mixtral 8x7B (Mistral)

3. **量化**:
   - SmoothQuant (MIT)
   - AWQ (MIT)
   - GPTQ (IST Austria)

4. **推理优化**:
   - vLLM (UC Berkeley)
   - SGLang (UC Berkeley)
   - TensorRT-LLM (NVIDIA)

### 10.2 代码参考

1. **FlashAttention**: https://github.com/Dao-AILab/flash-attention
2. **vLLM**: https://github.com/vllm-project/vllm
3. **SGLang**: https://github.com/sgl-project/sglang
4. **CUTLASS**: https://github.com/NVIDIA/cutlass

### 10.3 学习路径

```
Level 1 (基础):
- 理解 Transformer 架构
- 掌握 Attention 计算流程
- 了解 KV Cache 原理

Level 2 (进阶):
- 学习 FlashAttention 原理
- 理解 Paged Attention 设计
- 掌握 FP8 量化方法

Level 3 (高级):
- 深入 MoE 架构和优化
- 学习分布式推理 (TP/EP)
- 掌握 CUDA kernel 优化技巧

Level 4 (专家):
- 阅读 HPC-Ops 源码
- 理解硬件特性 (TMA, PDL, Multicast)
- 设计自己的优化方案
```

---

## 附录: 关键术语表

| 术语 | 解释 |
|------|------|
| **Prefill** | 处理整个 prompt 的阶段，计算密集型 |
| **Decode** | 逐 token 生成的阶段，访存密集型 |
| **KV Cache** | 缓存历史 token 的 Key 和 Value |
| **Paged Attention** | 将 KV Cache 分成固定大小的 block |
| **Split-K** | 将 K 维度切分到多个 CTA 并行计算 |
| **CTA** | Cooperative Thread Array，即 CUDA block |
| **TMA** | Tensor Memory Accelerator，SM90 的硬件内存访问单元 |
| **PDL** | Programmatic Dependent Launch，程序化依赖启动 |
| **FP8** | 8-bit 浮点数，包括 E4M3 和 E5M2 格式 |
| **MoE** | Mixture of Experts，混合专家模型 |
| **TP** | Tensor Parallelism，张量并行 |
| **EP** | Expert Parallelism，专家并行 |
| **AllReduce** | 分布式归约操作，汇总所有 GPU 的结果 |
| **Multicast** | 一次发送给多个 GPU 的通信方式 |
| **Lamport** | 无锁通信协议 |
| **SiLU** | SiLU(x) = x * sigmoid(x)，激活函数 |
| **SwiGLU** | SiLU + Gate 的变体，主流 LLM 选择 |

---

*本文档基于 HPC-Ops 项目代码分析，适合作为 LLM 算法实习面试的准备材料。建议结合代码阅读和实际实验加深理解。*

*最后更新: 2026 年 6 月*
