# HPC-Ops 8卡4090复现与学习计划

> 基于实际硬件资源（8x RTX 4090）制定的完整复现与学习方案

---

## 目录

1. [硬件能力评估与限制分析](#1-硬件能力评估与限制分析)
2. [复现策略总览](#2-复现策略总览)
3. [阶段一：环境搭建与基础验证（Week 1）](#3-阶段一环境搭建与基础验证week-1)
4. [阶段二：算法理解与PyTorch复现（Week 2-3）](#4-阶段二算法理解与pytorch复现week-2-3)
5. [阶段三：CUDA Kernel学习与实现（Week 4-5）](#5-阶段三cuda-kernel学习与实现week-4-5)
6. [阶段四：分布式推理与性能优化（Week 6）](#6-阶段四分布式推理与性能优化week-6)
7. [阶段五：对比验证与总结（Week 7）](#7-阶段五对比验证与总结week-7)
8. [附录：关键代码映射表](#附录关键代码映射表)

---

## 1. 硬件能力评估与限制分析

### 1.1 4090 vs H20 关键差异

| 特性 | RTX 4090 | H20 (Hopper) | 影响 |
|------|----------|--------------|------|
| **架构** | SM89 (Ada Lovelace) | SM90a (Hopper) | 指令集不兼容 |
| **显存** | 24 GB GDDR6X | 96 GB HBM3 | 大模型需要更多卡 |
| **FP8 Tensor Core** | ❌ 不支持 | ✅ 支持 | 无法复现FP8优化 |
| **BF16 Tensor Core** | ✅ 支持 | ✅ 支持 | 可以用BF16替代 |
| **TMA** | ❌ 不支持 | ✅ 支持 | 无法复现TMA路径 |
| **PDL** | ❌ 不支持 | ✅ 支持 | 无法复现PDL优化 |
| **Multicast** | ❌ 不支持 | ✅ 支持 | 无法复现AllReduce融合 |
| **cp.async** | ✅ 支持 | ✅ 支持 | 可以复现cp.async路径 |
| **NVLink** | ❌ 仅有PCIe | ✅ NVLink 900GB/s | 通信带宽差距大 |

### 1.2 可复现与不可复现的模块

**✅ 可以完整复现**：
- Python API 层（`hpc/*.py`）
- 算法逻辑（Attention、MoE、Sampler等）
- 测试框架（`tests/utils.py`、`conftest.py`）
- Benchmark 框架
- PyTorch 原生实现的参考代码

**⚠️ 部分可复现**：
- Group GEMM（cp.async路径，但没有FP8 MMA）
- Fused MoE（cp.async路径，用BF16替代FP8）
- Attention（用BF16，没有动态调度）

**❌ 无法复现**：
- FP8 量化相关（没有FP8 Tensor Core）
- TMA 路径的kernel
- PDL 优化的流水线
- Multicast AllReduce
- SM90 特有的 MMA 指令

### 1.3 复现策略调整

```
原策略: 直接编译运行 HPC-Ops (需要SM90)
    ↓
调整后: 
1. 用 PyTorch 原生实现理解算法
2. 学习 CUDA 优化核心概念
3. 用 BF16 替代 FP8 进行验证
4. 对比 HPC-Ops 和 PyTorch 的性能差异
5. 尝试在4090上实现简化版kernel
```

---

## 2. 复现策略总览

### 2.1 学习目标

**核心目标**：
1. **理解 LLM 推理的核心算子**：Attention、MoE、GEMM、Sampler
2. **掌握 CUDA 优化技巧**：Kernel Fusion、内存优化、并行策略
3. **理解分布式推理**：Tensor Parallelism、Expert Parallelism
4. **建立性能评估能力**：Benchmark设计、Profiling、瓶颈分析

**不要追求**：
- 在4090上完整复现HPC-Ops的所有优化
- 达到H20的性能水平

### 2.2 学习路径

```
Week 1: 环境搭建 + 理解项目架构
Week 2: Attention 机制深入学习
Week 3: MoE 架构 + 量化技术
Week 4: CUDA Kernel 基础 + 简单实现
Week 5: Kernel Fusion + 性能优化
Week 6: 分布式推理 + 8卡并行
Week 7: 对比验证 + 总结复盘
```

### 2.3 验证方法

**正确性验证**：
```python
# 对比 HPC-Ops 的 naive 实现和 PyTorch 实现
def test_attention():
    # 1. 用 PyTorch 原生实现
    output_pytorch = attention_pytorch(q, k, v)
    
    # 2. 用 HPC-Ops 的 naive 参考实现
    output_naive = attention_naive(q, k, v)
    
    # 3. 对比结果
    assert allclose(output_pytorch, output_naive, rtol=0.01, atol=0.01)
```

**性能验证**：
```python
# 测量不同实现的性能
import time

def benchmark(func, *args, warmup=3, repeat=100):
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(repeat):
        func(*args)
    torch.cuda.synchronize()
    return (time.time() - start) / repeat
```

---

## 3. 阶段一：环境搭建与基础验证（Week 1）

### 3.1 环境准备

**硬件需求**：
- 8x RTX 4090 (24GB each)
- 总显存: 192GB
- 足够跑 7B-13B 模型的推理

**软件环境**：
```bash
# 1. 创建 conda 环境
conda create -n hpc-ops python=3.10
conda activate hpc-ops

# 2. 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖
pip install transformers accelerate pytest numpy

# 4. 克隆项目
git clone https://github.com/Tencent/hpc-ops.git
cd hpc-ops
```

**注意**：不要尝试编译 HPC-ops 的 CUDA 代码（需要SM90），只使用 Python 层。

### 3.2 理解项目架构

**任务**：阅读并理解以下文件

```
1. README.md - 项目概述
2. hpc/__init__.py - Python API 入口
3. hpc/attention.py - Attention 算子接口
4. hpc/fuse_moe.py - Fused MoE 算子接口
5. hpc/sampler.py - Sampler 算子接口
6. tests/utils.py - 测试工具
```

**输出**：绘制项目架构图，理解数据流

### 3.3 运行参考实现

**任务**：运行 HPC-Ops 中的 naive 参考实现

```python
# tests/test_fuse_moe_pertensor.py 中的 naive 实现
def naive_gather_expert_inputs(x, topk_ids, num_expert, rank_ep):
    # 理解 token 如何按 expert 分组
    ...

def naive_group_gemm(x, w, cu_seqlens, scale, expert_ids):
    # 理解 grouped matrix multiplication
    ...

def naive_fuse_moe_pertensor_fp8(...):
    # 理解完整的 MoE 流程
    ...
```

**输出**：能解释每个 naive 函数的作用

---

## 4. 阶段二：算法理解与PyTorch复现（Week 2-3）

### 4.1 Attention 机制复现（Week 2）

**目标**：用纯 PyTorch 实现 Attention 的核心逻辑

**Day 1-2: 基础 Attention**
```python
# 实现标准 Attention
def attention_pytorch(q, k, v):
    """
    q: [batch, num_heads, seq_len, head_dim]
    k: [batch, num_kv_heads, seq_len, head_dim]
    v: [batch, num_kv_heads, seq_len, head_dim]
    """
    # 1. 计算 attention score
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # 2. Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    
    # 3. Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # 4. 加权求和
    output = torch.matmul(attn_weights, v)
    return output
```

**Day 3-4: KV Cache Attention**
```python
# 实现带 KV Cache 的 Attention
def attention_with_kvcache(q, k_cache, v_cache, k_new, v_new):
    """
    实现 paged kv cache 的逻辑
    """
    # 1. 更新 KV Cache
    k_cache = torch.cat([k_cache, k_new], dim=2)
    v_cache = torch.cat([v_cache, v_new], dim=2)
    
    # 2. 计算 Attention
    scores = torch.matmul(q, k_cache.transpose(-2, -1)) * scale
    output = torch.matmul(F.softmax(scores, dim=-1), v_cache)
    
    return output, k_cache, v_cache
```

**Day 5-7: Paged KV Cache**
```python
# 实现 Paged KV Cache 的核心逻辑
class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_heads, head_dim):
        self.block_size = block_size
        self.k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        self.v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        self.block_table = {}  # seq_id -> list of block_ids
        
    def allocate_block(self, seq_id):
        # 分配一个空闲 block
        ...
        
    def append_token(self, seq_id, k, v):
        # 追加一个 token 的 k, v
        ...
```

**验证**：
```python
# 对比 naive 实现和 PyTorch 实现
def test_attention():
    q = torch.randn(1, 32, 1, 128)  # [batch, heads, seq, dim]
    k = torch.randn(1, 32, 100, 128)
    v = torch.randn(1, 32, 100, 128)
    
    output_pytorch = attention_pytorch(q, k, v)
    output_naive = attention_naive(q, k, v)  # 从 HPC-Ops tests 中获取
    
    assert torch.allclose(output_pytorch, output_naive, rtol=0.01)
```

### 4.2 MoE 架构复现（Week 3）

**目标**：用纯 PyTorch 实现 Fused MoE 的核心逻辑

**Day 1-2: Router + Expert Selection**
```python
# 实现 MoE Router
class MoERouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k):
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        
    def forward(self, x):
        # x: [num_tokens, hidden_size]
        logits = self.gate(x)  # [num_tokens, num_experts]
        topk_logits, topk_ids = torch.topk(logits, self.top_k, dim=-1)
        topk_scale = F.softmax(topk_logits, dim=-1)
        return topk_ids, topk_scale
```

**Day 3-4: Expert GEMM**
```python
# 实现 Expert Parallel GEMM
def expert_gemm(x, weight, expert_ids, cu_seqlens):
    """
    x: [total_tokens, hidden_size]
    weight: [num_experts, intermediate_size, hidden_size]
    expert_ids: [total_tokens] - 每个 token 属于哪个 expert
    cu_seqlens: [num_experts + 1] - 每个 expert 的 token 起止位置
    """
    output = torch.zeros(total_tokens, intermediate_size)
    for i in range(num_experts):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        x_expert = x[start:end]
        output[start:end] = x_expert @ weight[i].T
    return output
```

**Day 5-7: 完整 Fused MoE**
```python
# 实现完整的 Fused MoE
def fused_moe_pytorch(x, router, gate_up_proj, down_proj, topk_ids, topk_scale):
    """
    完整的 MoE 前向传播
    """
    # 1. Token 分组 (count_and_gather)
    sorted_x, sorted_indices = gather_by_expert(x, topk_ids)
    
    # 2. Gate-Up Projection
    gate_up_output = expert_gemm(sorted_x, gate_up_proj, ...)
    
    # 3. Activation (SiLU) + 量化
    gate, up = gate_up_output.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    
    # 4. Down Projection
    down_output = expert_gemm(activated, down_proj, ...)
    
    # 5. 加权归约 (reduce)
    output = reduce_by_expert(down_output, topk_scale, sorted_indices)
    
    return output
```

**验证**：
```python
# 对比 PyTorch 实现和 HPC-Ops naive 实现
def test_fused_moe():
    # 使用 tests/test_fuse_moe_pertensor.py 中的测试数据
    x = torch.randn(num_seq, hidden_size, dtype=torch.float8_e4m3fn)
    ...
    
    output_pytorch = fused_moe_pytorch(...)
    output_naive = naive_fuse_moe_pertensor_fp8(...)
    
    assert allclose(output_pytorch, output_naive, rtol=0.08, atol=0.1)
```

---

## 5. 阶段三：CUDA Kernel学习与实现（Week 4-5）

### 5.1 CUDA 基础回顾（Week 4）

**目标**：掌握 CUDA 编程基础

**Day 1-2: 基础 Kernel**
```cuda
// 实现向量加法
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 实现矩阵乘法（基础版）
__global__ void matmul_basic(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**Day 3-4: 共享内存优化**
```cuda
// 使用共享内存优化矩阵乘法
#define TILE_SIZE 32

__global__ void matmul_shared(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载到共享内存
        As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        // 计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

**Day 5-7: cp.async 异步加载**
```cuda
// 使用 cp.async 实现异步数据加载
__global__ void matmul_cp_async(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // 使用 cp.async 异步加载数据
    // cp.async.cg.shared.global [dst], [src], size
    // 这允许计算和内存访问 overlap
    ...
}
```

### 5.2 HPC-Ops 核心 Kernel 学习（Week 5）

**目标**：理解 HPC-Ops 中的关键 Kernel 设计

**Day 1-2: RMSNorm Kernel**
```cuda
// 参考: src/normalization/fused_rmsnorm_with_scale.cu
// 实现 fused RMSNorm + FP8 量化

template <int kHiddenStates>
__global__ void fused_rmsnorm_kernel(
    const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight,
    __nv_fp8_e4m3 *output,
    float *output_scale,
    float eps
) {
    // 1. 计算 RMS
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < kHiddenStates; i += blockDim.x) {
        float val = __bfloat162float(input[blockIdx.x * kHiddenStates + i]);
        sum_sq += val * val;
    }
    // Warp reduce
    sum_sq = warp_reduce_sum(sum_sq);
    
    float rms = rsqrtf(sum_sq / kHiddenStates + eps);
    
    // 2. Normalize + Scale + Quantize
    for (int i = threadIdx.x; i < kHiddenStates; i += blockDim.x) {
        float val = __bfloat162float(input[blockIdx.x * kHiddenStates + i]);
        float weight_val = __bfloat162float(weight[i]);
        float normalized = val * rms * weight_val;
        
        // FP8 量化
        float scale = compute_fp8_scale(normalized);
        output[blockIdx.x * kHiddenStates + i] = float_to_fp8(normalized * scale);
    }
}
```

**Day 3-4: Attention Decode Kernel**
```cuda
// 参考: src/attention/decode/sm90/static/
// 实现简化版的 decode attention

// 核心思想: Split-K 并行
// 将 KV 序列切分到多个 CTA，每个 CTA 计算部分结果，最后 reduce

template <int kTileN>
__global__ void attention_decode_kernel(
    const float *q,      // [num_heads, head_dim]
    const float *k_cache, // [num_blocks, block_size, num_heads, head_dim]
    const float *v_cache,
    float *output,
    int *block_ids,
    int kv_len
) {
    // 1. 每个 CTA 处理一段 KV
    int kv_start = blockIdx.x * kTileN;
    int kv_end = min(kv_start + kTileN, kv_len);
    
    // 2. 计算 Q @ K^T
    float scores[kTileN];
    for (int i = 0; i < kv_end - kv_start; i++) {
        scores[i] = dot_product(q, k_cache + (kv_start + i) * head_dim);
    }
    
    // 3. Softmax
    float max_score = max_element(scores, kv_end - kv_start);
    float sum_exp = 0.0f;
    for (int i = 0; i < kv_end - kv_start; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }
    
    // 4. 加权求和 V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int i = 0; i < kv_end - kv_start; i++) {
            val += scores[i] * v_cache[(kv_start + i) * head_dim + d];
        }
        atomicAdd(output + d, val / sum_exp);
    }
}
```

**Day 5-7: Group GEMM Kernel**
```cuda
// 参考: src/group_gemm/cp_async/group_gemm_fp8.cu
// 实现简化版的 group gemm

// 核心思想: 多个 expert 的 GEMM 融合到一个 kernel
// 使用 task map 动态分配任务

template <int kTileM, int kTileN>
__global__ void group_gemm_kernel(
    const float *A,        // [total_tokens, K]
    const float *B,        // [num_experts, N, K]
    float *C,              // [total_tokens, N]
    const int *cu_seqlens, // [num_experts + 1]
    int K, int N
) {
    // 1. 从 task map 获取当前 CTA 处理的任务
    int expert_id = get_expert_from_task_map(blockIdx.x);
    int tile_m = get_tile_m_from_task_map(blockIdx.x);
    int tile_n = get_tile_n_from_task_map(blockIdx.x);
    
    // 2. 计算 GEMM
    int m_start = cu_seqlens[expert_id] + tile_m * kTileM;
    int n_start = tile_n * kTileN;
    
    float C_local[kTileM][kTileN] = {0};
    for (int k = 0; k < K; k += 32) {
        // 加载 A, B 到寄存器/共享内存
        // 计算 C_local += A_tile @ B_tile
    }
    
    // 3. 写回结果
    for (int i = 0; i < kTileM; i++) {
        for (int j = 0; j < kTileN; j++) {
            C[(m_start + i) * N + n_start + j] = C_local[i][j];
        }
    }
}
```

---

## 6. 阶段四：分布式推理与性能优化（Week 6）

### 6.1 Tensor Parallelism 实现

**目标**：在 8 卡 4090 上实现 Tensor Parallelism

**Day 1-2: Column Parallel Linear**
```python
# 实现 Column Parallel Linear
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, num_gpus):
        self.weight = nn.Parameter(torch.empty(out_features // num_gpus, in_features))
        
    def forward(self, x):
        # 每个 GPU 计算一部分输出
        return F.linear(x, self.weight)
```

**Day 3-4: Row Parallel Linear**
```python
# 实现 Row Parallel Linear
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, num_gpus):
        self.weight = nn.Parameter(torch.empty(out_features, in_features // num_gpus))
        
    def forward(self, x):
        # 每个 GPU 计算部分结果，然后 AllReduce
        partial_output = F.linear(x, self.weight)
        dist.all_reduce(partial_output)
        return partial_output
```

**Day 5-7: 完整的 TP 推理**
```python
# 实现完整的 Tensor Parallel 推理
def inference_with_tp(model, input_ids, num_gpus=8):
    # 1. 将模型切分到多个 GPU
    model = shard_model(model, num_gpus)
    
    # 2. 前向传播
    for layer in model.layers:
        # 每一层内部做 TP
        hidden_states = layer(hidden_states)
        # AllReduce 在每一层结束后
        dist.all_reduce(hidden_states)
    
    return hidden_states
```

### 6.2 性能 Profiling 与优化

**目标**：学习如何分析和优化性能

**Profiling 工具使用**：
```python
# 使用 PyTorch Profiler
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    model(input_ids)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**瓶颈分析**：
```python
# 分析 Attention 是 compute-bound 还是 memory-bound
def analyze_attention_bound(seq_len, head_dim, num_heads):
    compute = 2 * seq_len * seq_len * head_dim * num_heads  # FLOPs
    memory = (seq_len * head_dim * num_heads * 2) * 2  # bytes (Q + K + V)
    
    # 计算 arithmetic intensity
    ai = compute / memory
    
    # H20 的 FLOPS 和带宽
    flops = 148e12  # 148 TFLOPS (BF16)
    bandwidth = 4e12  # 4 TB/s
    
    roofline = flops / bandwidth
    
    if ai < roofline:
        return "Memory-bound"
    else:
        return "Compute-bound"
```

---

## 7. 阶段五：对比验证与总结（Week 7）

### 7.1 完整对比验证

**任务**：对比不同实现的正确性和性能

```python
# 完整的验证流程
def comprehensive_validation():
    # 1. Attention 验证
    test_attention_implementations()
    
    # 2. MoE 验证
    test_moe_implementations()
    
    # 3. 性能对比
    benchmark_all_implementations()
    
    # 4. 生成报告
    generate_validation_report()
```

### 7.2 总结复盘

**输出**：
1. 学习笔记：核心概念、关键设计、优化技巧
2. 代码仓库：自己实现的 CUDA kernel
3. 性能报告：不同实现的性能对比
4. 面试准备：能清晰讲解每个模块的设计和优化

---

## 附录：关键代码映射表

### A.1 Python API 与 Naive 实现对照

| 功能 | Python API | Naive 实现位置 |
|------|------------|----------------|
| Attention Prefill | `hpc/attention.py:15-67` | `tests/test_attention_prefill_bf16.py` |
| Attention Decode | `hpc/attention.py:341-412` | `tests/test_attention_decode_bf16.py` |
| Fused MoE | `hpc/fuse_moe.py:136-166` | `tests/test_fuse_moe_pertensor.py:119-150` |
| Sampler | `hpc/sampler.py:42-182` | `tests/test_sampler.py` |
| RMSNorm | `hpc/normalization.py` | `tests/test_normalization.py` |

### A.2 CUDA Kernel 与学习重点

| Kernel | 位置 | 学习重点 |
|--------|------|----------|
| RMSNorm | `src/normalization/fused_rmsnorm_with_scale.cu` | Warp Reduce、向量化加载 |
| Attention Decode | `src/attention/decode/sm90/` | Split-K、共享内存优化 |
| Group GEMM | `src/group_gemm/cp_async/` | cp.async、Task Map |
| Fused MoE | `src/fuse_moe/cp_async/` | Kernel Fusion、流水线 |

### A.3 测试文件与验证方法

| 测试文件 | 验证内容 | 关键参数 |
|----------|----------|----------|
| `test_fuse_moe_pertensor.py` | Fused MoE 正确性 | rtol=0.08, atol=0.1 |
| `test_attention_decode_bf16.py` | Decode Attention | 与 PyTorch 对比 |
| `test_sampler.py` | Sampler 正确性 | 分布一致性 |
| `test_normalization.py` | RMSNorm 正确性 | 误差分析 |

---

## 重要提醒

1. **不要编译 HPC-Ops 的 CUDA 代码**：需要 SM90，4090 无法运行
2. **专注于算法理解**：用 PyTorch 实现理解核心逻辑
3. **学习优化思想**：即使无法复现 SM90 优化，也要理解其设计思想
4. **建立性能直觉**：通过 Profiling 理解什么是 compute-bound vs memory-bound
5. **准备面试**：能清晰讲解每个模块的设计和优化

---

*本计划基于 8x RTX 4090 的实际硬件资源制定，旨在最大化学习效果而非完整复现 HPC-Ops。*

*最后更新: 2026 年 6 月*
