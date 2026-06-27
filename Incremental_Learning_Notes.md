# HPC-Ops 学习增量笔记

> 基于8卡4090复现计划的学习过程中，对比前两轮分析新增的理解和洞察

---

## 目录

1. [硬件架构深入理解](#1-硬件架构深入理解)
2. [CUDA编程核心概念](#2-cuda编程核心概念)
3. [Attention机制增量理解](#3-attention机制增量理解)
4. [MoE架构增量理解](#4-moe架构增量理解)
5. [量化技术增量理解](#5-量化技术增量理解)
6. [性能优化增量理解](#6-性能优化增量理解)
7. [分布式推理增量理解](#7-分布式推理增量理解)
8. [工程实践增量理解](#8-工程实践增量理解)

---

## 1. 硬件架构深入理解

### 1.1 SM90 vs SM89 架构差异

**前两轮理解**：
- 知道 H20 是 SM90，4090 是 SM89
- 知道 SM90 有 TMA、FP8 等新特性

**本轮增量理解**：

```
【TMA (Tensor Memory Accelerator) 深入理解】

之前理解: TMA 是一种硬件加速的内存访问方式
本轮理解: TMA 的核心价值在于"异步多维tensor加载"

具体机制:
1. TMA 可以异步加载 1D/2D/3D tensor，不需要 CPU 干预
2. 支持越界检测，可以安全加载不规则形状
3. 减少 warp 的内存访问负担，warp 只需要等待 TMA 完成

代码体现: src/utils/tma.cuh
```cuda
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
// TMA 指令只在 SM90 上可用
auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, tma_copy_layout_q);
#endif
```

为什么 4090 不能用:
- 4090 没有 TMA 硬件单元
- 只能用 cp.async 做异步加载，但 cp.async 是 1D 的
- 对于多维 tensor，需要手动计算地址，代码复杂且容易出错

【PDL (Programmatic Dependent Launch) 深入理解】

之前理解: PDL 可以让 kernel 在前一个 kernel 完成前启动
本轮理解: PDL 的核心是"减少 kernel launch bubble"

具体机制:
1. 传统 CUDA: kernel A 完成 -> launch kernel B -> kernel B 开始
2. PDL: kernel A 快完成时就 launch kernel B，overlap 执行

代码体现: src/fuse_moe/fuse_moe.cu
```cpp
bool use_pdl = true;
// ...
cudaGridDependencySynchronize();  // 等待依赖
cudaTriggerProgrammaticLaunchCompletion();  // 触发下一个 kernel
```

为什么 4090 不能用:
- PDL 需要 SM90 的硬件支持
- 4090 只能用传统的方式，kernel 之间无法 overlap

【Multicast 深入理解】

之前理解: Multicast 可以一次发送给多个 GPU
本轮理解: Multicast 的核心是"减少 NVLink 通信次数"

具体机制:
1. 传统 AllReduce: 每个 GPU 都需要单独发送数据
2. Multicast: 一次写入，所有 GPU 都能读到

代码体现: src/allreduce/fuse_allreduce_rmsnorm_high_throughput.cu
```cuda
// 使用 multicast buffer
T* broadcastBufW = reinterpret_cast<T*>(
    flag.getCurLamportBuf(mcastPtr, MNNVLTwoShotStage::BROADCAST));
```

为什么 4090 不能用:
- 4090 没有 NVLink，只有 PCIe
- PCIe 没有 multicast 能力
```

### 1.2 内存层次深入理解

**前两轮理解**：
- 知道 GPU 有 Global Memory、Shared Memory、Register
- 知道 Shared Memory 比 Global Memory 快

**本轮增量理解**：

```
【内存层次与性能关系】

Global Memory (HBM):
- 带宽: H20 4TB/s, 4090 1TB/s
- 延迟: ~400 cycles
- 大小: H20 96GB, 4090 24GB

Shared Memory (SMEM):
- 带宽: ~19TB/s (32 banks x 4B x 32 cycles)
- 延迟: ~20 cycles
- 大小: H20 228KB/SM, 4090 100KB/SM

Register:
- 带宽: 理论无限
- 延迟: 1 cycle
- 大小: 255 registers/thread

【关键洞察】

1. 为什么 Kernel Fusion 重要?
   - 每次 kernel launch 都需要从 HBM 读取数据
   - 融合后只需要读取一次，写回一次
   - Fused MoE: 5 个 kernel -> 1 个流水线，减少 4 次 HBM 读写

2. 为什么 Shared Memory 重要?
   - Attention 计算需要多次访问相同的数据
   - 用 Shared Memory 缓存，避免重复从 HBM 读取
   - 但 Shared Memory 有 bank conflict 问题

3. 为什么 Register 关键?
   - MMA 指令的操作数在 Register 中
   - Register 越多，可以保持更多的中间结果
   - 但 Register 太多会降低 occupancy

【代码体现】

HPC-Ops 的内存优化策略:
1. 使用 TMA 将数据从 HBM 异步加载到 SMEM
2. 从 SMEM 加载到 Register
3. 在 Register 中完成计算
4. 将结果从 Register 写回 HBM

关键设计: 减少 HBM 访问次数
```

---

## 2. CUDA编程核心概念

### 2.1 Warp 与 Warp Specialization

**前两轮理解**：
- 知道 Warp 是 32 个线程
- 知道 Warp Specialization 是将 Warp 分为不同角色

**本轮增量理解**：

```
【Warp Specialization 深入理解】

传统方式:
- 所有 Warp 执行相同的代码
- 每个 Warp 都需要做数据加载和计算
- 计算单元和内存单元都有空闲

Warp Specialization:
- Producer Warp: 只做数据加载
- Consumer Warp: 只做计算
- 通过 barrier 同步

代码体现: src/fuse_moe/fuse_moe.cu (TMA 路径)
```cuda
// Producer Warp: 负责 TMA 加载
if (warp_id < num_producer_warps) {
    // 循环: 加载数据到 SMEM
    tma_copy(Q, ...);
    tma_copy(K, ...);
    barrier.arrive();
}

// Consumer Warp: 负责计算
else {
    // 循环: 等待数据，执行 MMA
    barrier.wait();
    mma(Q_smem, K_smem, ...);
}
```

为什么 cp.async 路径不用 Warp Specialization:
- cp.async 的开销比 TMA 大
- 小尺寸时 Warp Specialization 的收益不明显
- 更高的 CTA 占用率更重要

【4090 的限制】
- 4090 没有 TMA，只能用 cp.async
- cp.async 是 1D 的，做 Warp Specialization 复杂
- 通常不做 Warp Specialization，用更简单的流水线
```

### 2.2 CTA 调度与负载均衡

**前两轮理解**：
- 知道 CTA 是 CUDA Block
- 知道负载均衡很重要

**本轮增量理解**：

```
【CTA 调度深入理解】

问题: Decode Attention 的负载不均衡

场景:
- 请求 A: KV 长度 10000 tokens
- 请求 B: KV 长度 100 tokens
- 静态 split-K=4:
  - 请求 A: 每个 split 处理 2500 tokens
  - 请求 B: 每个 split 处理 25 tokens
  - 负载不均衡!

【Greedy Bin-Packing 解决方案】

代码体现: hpc/attention.py:515-621

算法:
1. 将所有请求切成统一大小的 KV tile (64 tokens)
2. 按 tile 数量降序排列
3. 贪心地将 tile 分配到负载最小的 CTA

效果:
- 请求 A: 10000/64 = 156 tiles
- 请求 B: 100/64 = 2 tiles
- 如果有 100 个 CTA:
  - 每个 CTA 平均分配 1.58 tiles
  - 负载均衡度大幅提升

【Task Map 设计】

```cpp
struct TaskScheduleInfo {
    int ihead_kv;      // KV head 索引
    int ibatch;        // batch 索引
    int ichunk;        // chunk 索引
    int iseq_start;    // 序列起始位置
    int num_seqkv;     // KV 序列长度
    int num_tile_kv;   // KV tile 数量
    // ...
};
```

每个 Task 48 bytes，预分配避免运行时分配
```

### 2.3 Split-K 技术

**前两轮理解**：
- 知道 Split-K 是将 K 维度切分到多个 CTA
- 知道可以提高并行度

**本轮增量理解**：

```
【Split-K 深入理解】

问题: 当 K 很大时，单个 CTA 计算时间长

场景:
- Q: [1, 128] (decode 阶段只有 1 个 query)
- K: [10000, 128] (很长的 KV cache)
- 如果不分割: 1 个 CTA 计算 10000 次 dot product

Split-K 解决方案:
1. 将 K 切分为 4 个 chunk
2. 4 个 CTA 并行计算各自的 chunk
3. 最后 reduce 结果

代码实现: src/attention/decode/sm90/static/
```cuda
// Split-K kernel
// 每个 CTA 处理 [kv_start, kv_end) 范围的 KV
int kv_start = blockIdx.x * kTileN;
int kv_end = min(kv_start + kTileN, kv_len);

// 计算部分结果
float partial_output[head_dim];
for (int i = kv_start; i < kv_end; i++) {
    float score = dot(q, k_cache[i]);
    // accumulate
}

// Combine kernel
// 将多个 split-K 的结果合并
// 使用 LSE (Log Sum Exp) 做加权平均
```

【LSE Combine 技术】

问题: 如何合并多个 split-K 的结果?

传统方法: 直接平均
- 问题: 不同 chunk 的 softmax 分母不同

LSE 方法:
1. 每个 chunk 保存: output 和 lse (log sum exp)
2. Combine 时: 使用 lse 做加权平均

```python
# combine kernel 的逻辑
def combine_splitk(outputs, lses):
    # lse[i] = log(sum(exp(scores_i)))
    # 最终输出 = sum(output[i] * exp(lse[i] - lse_max)) / sum(exp(lse[i] - lse_max))
    lse_max = max(lses)
    weights = [exp(lse - lse_max) for lse in lses]
    total_weight = sum(weights)
    final_output = sum(w * o for w, o in zip(weights, outputs)) / total_weight
    return final_output
```
```

---

## 3. Attention机制增量理解

### 3.1 KV Cache 管理

**前两轮理解**：
- 知道 KV Cache 缓存历史 K, V
- 知道 Paged Attention 用 block 管理

**本轮增量理解**：

```
【KV Cache 内存布局深入理解】

NHD 布局:
- [num_blocks, block_size, num_heads, dim]
- 优点: 连续访问同一个 block 的所有 heads
- 缺点: 不同 heads 之间有 stride

HND 布局:
- [num_heads, num_blocks, block_size, dim]
- 优点: 连续访问同一个 head 的所有 blocks
- 缺点: 不同 blocks 之间有 stride

代码体现: hpc/attention.py:70-145
```python
# 支持两种布局，通过 stride 区分
# kcache: [num_blocks, block_size, num_head_kv, dim]
# 如果是 HND 布局，stride 会不同
ldK = kcache.stride(0)
ldK1 = kcache.stride(1)
ldK2 = kcache.stride(2)
```

【Block Size 选择】

问题: Block Size 设为多少合适?

太小 (如 4):
- 优点: 内存利用率高
- 缺点: page table 管理开销大

太大 (如 64):
- 优点: 管理简单
- 缺点: 内部碎片多

经验值: 16 tokens
- 平衡了管理开销和内存效率
- 与 GPU 的 warp size 对齐

【跨请求共享 KV Cache】

场景: 多个请求共享相同的 system prompt
- system prompt 的 KV Cache 可以只存一份
- 不同请求通过 block table 引用同一份 KV

代码体现: block_ids 参数
```python
block_ids: Tensor  # [num_batch, max_blocks]
# 每个请求有自己的 page table
# 不同请求可以指向相同的物理 block
```
```

### 3.2 FP8 Attention 计算

**前两轮理解**：
- 知道 FP8 可以减半内存和计算
- 知道有 per-tensor 和 per-token 两种量化

**本轮增量理解**：

```
【FP8 Attention 计算细节】

公式:
Attention = softmax(Q_fp8 * K_fp8^T * qscale * kscale / sqrt(d_k)) * V_fp8 * vscale

关键问题: scale 如何传递?

1. Q 的 scale:
   - Per-token per-head: qscale [batch, num_heads, seq_len]
   - 每个 token 的每个 head 有独立的 scale
   
2. K 的 scale:
   - Per-tensor (quant_type=1): kscale [1]
   - Per-token per-head (quant_type=0): kscale [num_blocks, ...]

3. V 的 scale:
   - 通常用 per-tensor: vscale [1]

【误差累积分析】

假设:
- Q 量化误差: ε_q ≈ 1%
- K 量化误差: ε_k ≈ 1%

Q @ K^T 的误差:
- 理论上: ε_{qk} ≈ ε_q + ε_k ≈ 2%
- 实际上: 误差会累积，可能更大

Softmax 的影响:
- Softmax 会放大差异
- 如果 Q @ K^T 的误差导致排序变化，影响很大

【为什么用 per-token 量化】

Per-tensor 量化的问题:
- 整个 tensor 用一个 scale
- 如果有异常值，scale 会很大
- 正常值的精度损失大

Per-token 量化的优势:
- 每个 token 独立的 scale
- 异常值只影响自己那个 token
- 其他 token 保持高精度

代码体现: hpc/act.py
```python
def scaled_fp8_quant(x: Tensor, scale: Tensor) -> Tensor:
    # x: [num_tokens, dim]
    # scale: [num_tokens, 1]  # per-token
    return (x / scale).to(torch.float8_e4m3fn)
```
```

---

## 4. MoE架构增量理解

### 4.1 Token 路由与负载均衡

**前两轮理解**：
- 知道 Router 选择 top-k experts
- 知道有负载均衡问题

**本轮增量理解**：

```
【Router 精度敏感性深入理解】

为什么 Router 对精度敏感?

1. Router 的输出是离散的 expert 选择
   - 微小的 logit 变化可能导致 top-k 选择完全不同
   - 不同 expert 处理结果差异大，选择错误影响质量

2. 加权系数 (topk_scale) 影响最终输出
   - output = sum(topk_scale[i] * expert[i](input))
   - 如果 topk_scale 不准确，输出会有偏差

【BF16 x FP32 GEMM 的设计动机】

代码体现: hpc/gemm.py

问题: Router GEMM 需要高精度
- 权重用 FP32 保持精度
- 但 FP32 Tensor Core 吞吐低 (4090 没有原生 FP32 TC)

解决方案: 分解为两个 BF16 GEMM
```python
scale = 1 / 256
w_high = w_fp32.to(torch.bfloat16)
w_low = ((w_fp32 - w_high.float()) / scale).to(torch.bfloat16)
# Result = x @ w_high + scale * (x @ w_low)
```

为什么 scale 用 1/256:
- 256 = 2^8，可以用位移，开销小
- 保证 w_low 在 BF16 范围内
- 两个 BF16 GEMM 的结果精度接近 FP32
```

### 4.2 Fused MoE 流水线

**前两轮理解**：
- 知道 Fused MoE 融合了多个步骤
- 知道可以减少 kernel launch 开销

**本轮增量理解**：

```
【Fused MoE 流水线深入理解】

传统流程 (5 个 kernel):
1. count_and_gather: Token 排序
2. group_gemm: Gate-Up GEMM
3. activation: SiLU + 量化
4. group_gemm: Down GEMM
5. reduce: 加权归约

每个 kernel 都需要:
- 从 HBM 读取输入
- 计算
- 写回 HBM

Fused 流程 (1 个流水线):
1. count_and_gather -> 结果在 SMEM/寄存器
2. group_gemm -> 直接使用，不需要读 HBM
3. activation -> 直接使用
4. group_gemm -> 直接使用
5. reduce -> 写回 HBM

内存访问减少:
- 传统: 5 次读 + 5 次写 = 10 次 HBM 访问
- Fused: 1 次读 + 1 次写 = 2 次 HBM 访问
- 减少 80% 的 HBM 访问!

【PDL 如何帮助 Fused MoE】

代码体现: src/fuse_moe/fuse_moe.cu

```cpp
bool use_pdl = true;

// Gate-Up GEMM 完成前，就可以启动 activation
// 因为 PDL 允许 kernel overlap
cudaGridDependencySynchronize();
// ... Gate-Up GEMM 的最后部分 ...
cudaTriggerProgrammaticLaunchCompletion();
// activation 可以提前开始
```

效果:
- Gate-Up GEMM 和 activation overlap
- activation 和 Down GEMM overlap
- 总时间 < 串行执行时间
```

### 4.3 TMA vs cp.async 路径选择

**前两轮理解**：
- 知道 TMA 更快，但需要 SM90
- 知道 cp.async 兼容性更好

**本轮增量理解**：

```
【路径选择的工程考量】

为什么 HPC-Ops 提供两条路径?

1. TMA 路径 (SM90 专用):
   - 使用 SM90_TMA_LOAD/SM90_TMA_STORE
   - 支持多维 tensor 异步加载
   - 可以做 Warp Specialization
   - 适合大 intermediate_size (>512)

2. cp.async 路径 (SM80+ 通用):
   - 使用 cp.async.cg.shared.global 指令
   - 只支持 1D 异步加载
   - 不做 Warp Specialization，CTA 占用率更高
   - 适合小 intermediate_size (<=512)

【阈值选择: 512】

代码体现: hpc/fuse_moe.py:133
```python
_CP_ASYNC_N_TP_MAX = 512
```

为什么是 512?

实验方法:
1. 扫描不同 intermediate_size: 256, 384, 512, 640, 768
2. 对比 TMA 和 cp.async 的性能
3. 找到性能交叉点

结果:
- intermediate_size <= 512: cp.async 更快
- intermediate_size > 512: TMA 更快

原因分析:
- 小尺寸时，TMA 的配置开销占比大
- cp.async 更简单，CTA 占用率更高
- 大尺寸时，TMA 的异步优势体现出来

【4090 上的策略】

4090 只能用 cp.async 路径:
- intermediate_size 通常 <= 2048
- 可以用 cp.async 实现类似的功能
- 但没有 TMA 的异步优势
- 需要手动管理数据加载
```

---

## 5. 量化技术增量理解

### 5.1 量化粒度选择

**前两轮理解**：
- 知道有 per-tensor、per-token、per-head 三种粒度
- 知道精度和开销的 trade-off

**本轮增量理解**：

```
【量化粒度深入理解】

1. Per-tensor 量化:
   - Scale shape: [1]
   - 整个 tensor 一个 scale
   - 最简单，但精度损失大
   
   适用场景: 
   - 对精度不敏感的场景
   - 权重量化 (权重分布相对稳定)

2. Per-token 量化:
   - Scale shape: [num_tokens]
   - 每个 token 一个 scale
   - 精度更好，开销适中
   
   适用场景:
   - Activation 量化
   - 不同 token 的动态范围差异大

3. Per-head 量化:
   - Scale shape: [num_heads]
   - 每个 head 一个 scale
   - 精度最好，开销大
   
   适用场景:
   - KV Cache 量化
   - 不同 head 的分布差异大

4. Per-token per-head 量化:
   - Scale shape: [num_tokens, num_heads]
   - 最精细，开销最大
   
   适用场景:
   - 对精度要求最高的场景
   - Q 的量化 (Q 的动态范围大)

【HPC-Ops 的量化策略选择】

代码体现: hpc/attention.py:8-12

```python
class QuantType(Enum):
    QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD = 0
    QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR = 1
```

默认使用 quant_type=1:
- Q: per-token per-head (精度敏感)
- K: per-tensor (节省 KV Cache 空间)
- V: per-tensor (节省 KV Cache 空间)

为什么 K/V 用 per-tensor:
- K/V 存在 KV Cache 中，需要长期保存
- Per-token scale 需要额外存储，增加内存
- K/V 的动态范围比 Q 更稳定

为什么 Q 用 per-token per-head:
- Q 每次重新计算，不需要长期保存
- Q 的动态范围可能很大
- Per-token per-head 可以保持高精度
```

### 5.2 FP8 精度损失分析

**前两轮理解**：
- 知道 FP8 有精度损失
- 知道可以通过量化策略缓解

**本轮增量理解**：

```
【精度损失的来源】

1. 截断误差:
   - FP8 E4M3 范围: [-448, 448]
   - 如果值超出范围，会被截断
   - 截断会导致很大的误差

2. 舍入误差:
   - FP8 只有 3 位尾数
   - 精度约为 2^-3 = 0.125
   - 相对误差约为 1%

3. 累积误差:
   - Attention 计算: Q @ K^T @ V
   - 每一步都有误差，误差会累积
   - 最终误差可能是单步误差的数倍

【如何评估精度损失】

代码体现: tests/utils.py

```python
def calculate_errors(ref_tensor, real_tensor):
    abs_error = torch.abs(ref_tensor - real_tensor)
    mae = torch.mean(abs_error).item()
    rel_error = abs_error / (torch.max(torch.abs(ref_tensor), torch.abs(real_tensor)) + eps)
    mre = torch.mean(rel_error).item()
    return {"mean_abs_error": mae, "mean_rel_error": mre}
```

评估指标:
1. MAE (Mean Absolute Error): 平均绝对误差
2. MRE (Mean Relative Error): 平均相对误差
3. Top-K 误差: 最大的 K 个误差

【可接受的精度范围】

经验值:
- MAE < 0.01: 优秀
- MAE < 0.05: 良好
- MAE < 0.1: 可接受
- MAE > 0.1: 需要优化

HPC-Ops 的阈值:
- rtol=0.08, atol=0.1
- 这是经过大量实验验证的阈值
- 对 LLM 生成质量影响可忽略
```

---

## 6. 性能优化增量理解

### 6.1 Kernel Fusion 的收益边界

**前两轮理解**：
- 知道 Kernel Fusion 可以减少开销
- 知道不是融合越多越好

**本轮增量理解**：

```
【Kernel Fusion 的 Trade-off】

收益:
1. 减少 kernel launch 开销 (~10us per kernel)
2. 减少 HBM 读写次数
3. 减少同步等待

成本:
1. Register 使用增加
2. Occupancy 可能降低
3. 代码复杂度增加

【收益边界分析】

场景 1: 小 kernel 融合
- 例: 两个 10us 的 kernel
- 融合后: 15us
- 收益: 5us (33%)
- 结论: 值得融合

场景 2: 大 kernel 融合
- 例: 两个 100us 的 kernel
- 融合后: 150us
- 收益: 50us (33%)
- 但 Register 使用增加，Occupancy 降低
- 结论: 需要评估

场景 3: 不兼容的 kernel 融合
- 例: compute-bound 和 memory-bound 的 kernel
- 融合后: 无法 overlap
- 收益: 很小
- 结论: 不值得融合

【HPC-Ops 的 Fusion 策略】

Fused MoE: 融合 5 个 kernel
- 为什么值得?
  - 中间数据可以保持在 SMEM/寄存器
  - 减少 80% 的 HBM 访问
  - PDL 可以 overlap 不同阶段

Fused Sampler: 融合成 2 个 kernel (不是 1 个)
- 为什么不融合成 1 个?
  - Top-K 和 Top-P 需要不同的并行策略
  - 强行融合会增加复杂度，收益不大
  - 2 个 kernel 是最优平衡点
```

### 6.2 内存访问模式优化

**前两轮理解**：
- 知道合并访问 (coalesced access) 很重要
- 知道 Shared Memory 可以减少 Global Memory 访问

**本轮增量理解**：

```
【内存访问模式深入理解】

1. 合并访问 (Coalesced Access):
   - 同一 Warp 的 32 个线程访问连续的地址
   - 硬件可以合并成 1-4 次内存事务
   - 如果访问不连续，事务数增加

2. Bank Conflict:
   - Shared Memory 分成 32 个 bank
   - 如果多个线程访问同一个 bank，会冲突
   - 需要通过 padding 避免

3. Swizzle:
   - HPC-Ops 使用 swizzle 避免 bank conflict
   - 通过改变地址映射，分散访问

代码体现: src/group_gemm/cp_async/group_gemm_fp8.cu

```cuda
// Swizzle layout
auto slayout_q = tile_to_shape(
    GMMA::Layout_K_SW128_Amon<Tin>{},  // SW128 = 128-bit swizzle
    make_shape(Int<kTileM>{}, Int<kTileK>{})
);
```

【4090 上的优化策略】

4090 的内存层次:
- HBM: 1TB/s
- L2 Cache: 72MB
- Shared Memory: 100KB/SM
- Register: 255/thread

优化重点:
1. 利用 L2 Cache (4090 的 L2 很大)
2. 减少 HBM 访问
3. 避免 Bank Conflict
4. 使用向量化加载 (128-bit)
```

### 6.3 Profiling 工具使用

**前两轮理解**：
- 知道可以用 Profiler 分析性能
- 知道要关注 kernel 时间

**本轮增量理解**：

```
【Profiling 深入理解】

1. PyTorch Profiler:
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

可以获取:
- 每个 kernel 的执行时间
- 内存分配情况
- kernel 的 shape 信息

2. Nsight Systems:
```bash
nsys profile --trace=cuda,nvtx python3 inference.py
```

可以获取:
- Timeline 视图
- kernel 的 overlap 情况
- 通信和计算的 overlap

3. Nsight Compute:
```bash
ncu --set full python3 inference.py
```

可以获取:
- 每个 kernel 的详细指标
- SM 占用率
- 内存带宽利用率
- Register 使用情况

【瓶颈分析方法】

1. Compute-bound 还是 Memory-bound?
   - 计算 Arithmetic Intensity (AI)
   - AI = FLOPs / Bytes
   - 对比 GPU 的 FLOPS/带宽比

2. Occupancy 分析:
   - 查看 Register 使用
   - 查看 Shared Memory 使用
   - 计算理论 Occupancy

3. 内存带宽分析:
   - 查看 HBM 读写字节数
   - 对比理论带宽
   - 找到带宽瓶颈
```

---

## 7. 分布式推理增量理解

### 7.1 Tensor Parallelism 实现

**前两轮理解**：
- 知道 TP 将权重切分到多个 GPU
- 知道需要 AllReduce 通信

**本轮增量理解**：

```
【TP 的通信模式】

1. Column Parallel:
   - 输入: 完整的 x
   - 权重: 每个 GPU 持有一部分 W
   - 输出: 每个 GPU 计算一部分 y
   - 通信: 无 (输出已经分片)

2. Row Parallel:
   - 输入: 每个 GPU 持有一部分 x
   - 权重: 每个 GPU 持有一部分 W
   - 输出: 每个 GPU 计算部分结果
   - 通信: AllReduce 汇总结果

【TP 的典型模式】

Transformer Layer:
1. Attention:
   - Q, K, V: Column Parallel (每个 GPU 持有部分 heads)
   - Output: Row Parallel + AllReduce

2. FFN:
   - Up/Gate: Column Parallel
   - Down: Row Parallel + AllReduce

通信量:
- 每层 2 次 AllReduce
- 通信量 = 2 * hidden_size * seq_len * batch_size

【4090 上的 TP 实现】

4090 没有 NVLink，只有 PCIe:
- PCIe 带宽: ~32GB/s (双向)
- NVLink 带宽: ~900GB/s (H20)
- 差距: ~28x

优化策略:
1. 减少通信量: 梯度压缩、量化通信
2. 通信计算 overlap: 在等待通信时做计算
3. 使用更快的互联: 如 InfiniBand (如果有)
```

### 7.2 Expert Parallelism 实现

**前两轮理解**：
- 知道 EP 将不同 expert 分配到不同 GPU
- 知道有 All-to-All 通信

**本轮增量理解**：

```
【EP 的通信模式】

1. Token 发送:
   - 每个 GPU 需要将 token 发送到对应的 expert 所在 GPU
   - 通信模式不规则，难以优化

2. Expert 计算:
   - 每个 GPU 只计算分配到的 experts
   - 计算量不均衡 (热门 expert 更多 token)

3. Token 接收:
   - 每个 GPU 需要接收处理完的 token
   - 通信模式同样不规则

【All-to-All 通信优化】

问题: All-to-All 通信模式不规则，难以优化

解决方案:
1. Token Dropping:
   - 丢弃需要跨节点通信的 token
   - 减少通信量，但损失精度

2. Expert Replication:
   - 在多个 GPU 上复制热门 expert
   - 减少跨节点通信，但增加内存

3. Hierarchical EP:
   - 先节点内 EP，再节点间 EP
   - 减少跨节点通信

【4090 上的 EP 实现】

8 卡 4090 的拓扑:
- 每台机器 8 卡，通过 PCIe 连接
- 没有 NVLink，通信带宽有限

优化策略:
1. 优先使用 TP 而不是 EP
2. 如果必须用 EP，尽量在节点内
3. 使用量化通信减少带宽需求
```

---

## 8. 工程实践增量理解

### 8.1 测试体系

**前两轮理解**：
- 知道 HPC-Ops 有完善的测试
- 知道用 pytest 做测试

**本轮增量理解**：

```
【测试体系深入理解】

1. 单元测试:
   - 26 个测试文件，覆盖所有算子
   - 每个测试用例测试一个功能点

2. 参数化测试:
```python
@pytest.mark.parametrize("num_seq", [128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [512])
...
def test_fuse_moe_pertensor_fp8(...):
    ...
```
   - 覆盖不同配置
   - 自动化测试

3. 精度验证:
```python
assert allclose(gt.to(torch.float32), my.to(torch.float32), rtol=0.08, atol=0.1)
```
   - 对比 naive 实现
   - 使用 MAE, MRE, Top-K 误差分析

4. Sanitizer 集成:
```python
# conftest.py
export SANITIZER_CHECK=memcheck,synccheck,racecheck
```
   - memcheck: 内存越界
   - synccheck: 同步错误
   - racecheck: 竞态条件

【4090 上的测试策略】

1. 使用 PyTorch 原生实现作为参考
2. 对比 HPC-Ops 的 naive 实现
3. 使用 4090 上的 CUDA kernel 实现
4. 验证正确性和性能
```

### 8.2 性能 Benchmark

**前两轮理解**：
- 知道 HPC-Ops 有 benchmark
- 知道要对比不同实现

**本轮增量理解**：

```
【Benchmark 设计原则】

1. 公平对比:
   - 相同的输入数据
   - 相同的硬件环境
   - 相同的测试方法

2. 预热:
```python
for _ in range(warmup):
    func(*args)
torch.cuda.synchronize()
```
   - 避免首次编译开销
   - 避免 CUDA context 初始化开销

3. 多次测量:
```python
for _ in range(repeat):
    func(*args)
torch.cuda.synchronize()
```
   - 取平均值或中位数
   - 避免偶然因素

4. 统计分析:
   - 计算均值、方差、百分位数
   - 识别异常值

【HPC-Ops 的 Benchmark 框架】

代码体现: benchmark/fused_moe/benchmark_fuse_moe.py

```python
MODEL_PRESETS = {
    "qwen3-235b":  (128, 8, 4096, 1536),
    "hunyuan-v3":  (192, 8, 4096, 1536),
    "deepseek-v3": (256, 8, 7168, 2048),
}

# 对比不同后端
ALL_BACKENDS = ["hpcops", "sglang", "vllm", "vllm_cutlass"]
```

设计特点:
1. 支持多个模型预设
2. 支持多个后端对比
3. 自动生成性能报告
```

---

## 持续更新

本笔记将随着学习进度持续更新，记录增量新学到的点。

**更新频率**: 每周至少更新一次

**更新内容**:
1. 新理解的概念
2. 新发现的设计细节
3. 新学到的优化技巧
4. 新遇到的问题和解决方案

---

*本笔记基于 8x RTX 4090 的复现计划，记录学习过程中的增量理解。*

*最后更新: 2026 年 6 月*
