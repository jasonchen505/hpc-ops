# LLM 算法实习技术面试应对指南

> 基于 HPC-Ops 项目深度分析，针对五类核心面试能力的系统化准备

---

## 目录

1. [第一类：底层原理深入理解](#1-第一类底层原理深入理解)
2. [第二类：实验和方案验证能力](#2-第二类实验和方案验证能力)
3. [第三类：问题定位能力](#3-第三类问题定位能力)
4. [第四类：工程落地能力](#4-第四类工程落地能力)
5. [第五类：业务与实际场景的理解](#5-第五类业务与实际场景的理解)
6. [综合模拟面试问答](#6-综合模拟面试问答)

---

## 1. 第一类：底层原理深入理解

### 1.1 核心考察点

面试官想看到的：
- **不只是背概念**，而是理解"为什么这样设计"
- 能讲清楚**解决什么问题**、**有什么局限**、**如何改进**

### 1.2 典型问题与深度回答

#### Q1: 为什么 LLM 推理需要 Paged KV Cache？解决了什么问题？

**表面回答**（大多数候选人会这样说）：
> Paged KV Cache 将 KV Cache 分成固定大小的 block，类似操作系统的虚拟内存，避免内存碎片。

**深度回答**（展示真正理解）：

```
【解决的核心问题】
传统 KV Cache 有三个致命问题：

1. 内存预分配浪费：
   - 每个请求必须预分配 max_seq_len 的连续内存
   - 实际生成长度通常远小于 max_seq_len
   - 128K 上下文时，浪费率可达 80%+

2. 内存碎片化：
   - 不同请求生成长度不同，释放后留下不规则空洞
   - 长时间服务后，物理内存有空间但无法分配
   - 类似操作系统的外部碎片问题

3. 并发动态批处理困难：
   - 不同请求的 KV Cache 长度不同
   - 连续内存布局下无法高效地做 batch 计算

【Paged Attention 如何解决】
参考代码: hpc/attention.py:70-145

```python
# block_ids: [num_batch, max_blocks] - page table
# kcache: [num_blocks, block_size, num_head_kv, dim]
```

- 将 KV Cache 切成固定大小的 block (通常 16 tokens)
- 通过 page table (block_ids) 管理映射关系
- 按需分配 block，用完即释放
- 支持跨请求共享 block (如 system prompt)

【局限性】
1. Page table 本身有管理开销 (内存 + 查找时间)
2. Block 内部仍有内部碎片 (最后一个 block 可能只用部分)
3. 需要额外的 block_ids tensor 传递，增加 API 复杂度
4. 不同 block_size 对性能有影响，需要调优

【改进方向】
1. 分层 Page Table: 支持更大的地址空间
2. Block size 自适应: 根据请求特征动态调整
3. 预取优化: 预测下一步需要的 block，提前加载
```

#### Q2: FP8 量化为什么对 LLM 推理重要？存在哪些精度风险？

**深度回答**：

```
【为什么重要】
1. 内存减半: FP8 (1B) vs BF16 (2B)
   - 128K 上下文的 KV Cache: 16GB (BF16) -> 8GB (FP8)
   - H20 GPU 96GB 显存，这 8GB 节省至关重要

2. 计算加速: H20 的 FP8 Tensor Core 吞吐是 BF16 的 2x
   - Attention 计算: Q@K^T 可以用 FP8 MMA
   - GEMM 计算: 权重用 FP8，计算量减半

3. 带宽优化: HBM 带宽是瓶颈，数据量减半 = 速度翻倍

【精度风险分析】
参考代码: hpc/attention.py:148-250

```python
class QuantType(Enum):
    QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD = 0  # 最高精度
    QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR = 1       # 平衡选择
```

风险点:
1. 动态范围受限: FP8 E4M3 范围 [-448, 448]，异常值会被截断
2. 量化误差累积: Attention = softmax(Q@K^T)@V
   - Q 有量化误差 ε_q, K 有 ε_k
   - Q@K^T 的误差 ≈ ε_q * K + Q * ε_k
   - softmax 会放大这个误差
3. 不同层的敏感度不同:
   - 浅层 feature 分布更均匀，量化损失小
   - 深层 feature 可能有长尾分布，量化损失大

【如何缓解】
1. Per-token 量化: 自适应每个 token 的动态范围
   - qscale shape: [num_batch, num_head, max_seq] 
   - 每个 token 独立的 scale
2. 混合精度: 敏感层用 BF16，不敏感层用 FP8
3. 量化感知训练 (QAT): 训练时模拟量化误差
4. SmoothQuant: 对 activation 做 per-channel scaling
```

#### Q3: MoE 中的路由 (Router) 为什么对精度敏感？

**深度回答**：

```
【Router 的作用】
Router 决定每个 token 由哪些 expert 处理:
gate_logits = x @ W_router  # [num_tokens, num_experts]
topk_ids = topk(gate_logits, k=8)
topk_scale = softmax(gate_logits[topk_ids])

【为什么精度敏感】
1. Router 的输出是离散的 expert 选择
   - 微小的 logit 变化可能导致 top-k 选择完全不同
   - 不同 expert 处理结果差异大，选择错误影响质量

2. Router GEMM 需要高精度:
   - 参考: hpc/gemm.py - BF16 x FP32 GEMM
   - Router 权重用 FP32，输入用 BF16
   - 分解为两个 BF16 GEMM 保持精度

3. Softmax 的数值稳定性:
   - topk_scale 是加权系数
   - 如果 softmax 数值不稳定，权重分配会出错

【HPC-Ops 的解决方案】
```python
# hpc/gemm.py - BF16 x FP32 GEMM
scale = 1 / 256
w_high = w_fp32.to(torch.bfloat16)
w_low = ((w_fp32 - w_high.float()) / scale).to(torch.bfloat16)
# Result = x @ w_high + scale * (x @ w_low)
```

为什么 scale 用 1/256?
- 256 = 2^8，可以用位移，开销小
- 保证 w_low 在 BF16 范围内
- 两个 BF16 GEMM 的结果精度接近 FP32

【局限性】
1. 需要两个 GEMM，计算量增加
2. 中间结果需要在寄存器中保持，增加 register pressure
3. 只对权重是 FP32 的场景有效
```

### 1.3 回答框架

回答底层原理问题时，建议使用这个框架：

```
1. 解决什么问题？ (Why)
   - 痛点是什么？没有这个方案会怎样？

2. 怎么解决的？ (How)
   - 核心思想是什么？
   - 关键设计决策是什么？

3. 有什么局限？ (Limitation)
   - 什么场景下效果不好？
   - 有什么 trade-off？

4. 如何改进？ (Improvement)
   - 业界有哪些改进方向？
   - 如果你来做，会怎么优化？
```

---

## 2. 第二类：实验和方案验证能力

### 2.1 核心考察点

面试官想看到的：
- 不只是"我做了什么"，而是**怎么证明有效**
- 能讲清楚**实验设计**、**对比基线**、**评估指标**、**结果分析**

### 2.2 典型问题与深度回答

#### Q4: 你们怎么验证 Fused MoE 的性能提升？

**深度回答**：

```
【实验设计】
参考: benchmark/fused_moe/benchmark_fuse_moe.py

1. 测试场景覆盖:
   - 不同模型: DeepSeek-V3, Hunyuan-V3, Qwen3-235B
   - 不同并行: TP=8 EP=1 vs TP=1 EP=8
   - 不同 batch size: 4, 16, 64, 128, 256, 512, 1024, 2048

2. 对比基线:
   - vLLM CUTLASS: 业界主流实现
   - vLLM Triton: Triton 实现
   - SGLang: 另一个推理框架

3. 评估指标:
   - 延迟 (Latency): 单请求端到端时间
   - 吞吐 (Throughput): tokens/second
   - 内存占用: peak memory usage

【关键代码】
```python
# benchmark/fused_moe/benchmark_fuse_moe.py
MODEL_PRESETS = {
    "qwen3-235b":  (128, 8, 4096, 1536),
    "hunyuan-v3":  (192, 8, 4096, 1536),
    "deepseek-v3": (256, 8, 7168, 2048),
}
```

【结果分析】
性能提升原因分析:
1. 减少 kernel launch: 4 个 kernel 融合成 1 个流水线
2. 减少内存访问: 中间结果不落地
3. PDL overlap: Gate-Up GEMM 和 activation overlap

精度验证:
```python
# tests/test_fuse_moe_pertensor.py
assert allclose(gt.to(torch.float32), my.to(torch.float32), rtol=0.08, atol=0.1)
```

- 对比 naive PyTorch 实现
- 使用 MAE, MRE, Top-K 误差分析
- rtol=0.08, atol=0.1 是可接受的精度范围

【追问准备】
Q: 为什么 rtol 设为 0.08？
A: 
- FP8 量化本身有约 1% 的误差
- 多次 GEMM 累积，误差会放大
- 0.08 的 rtol 对 LLM 生成质量影响可忽略
- 过于严格的阈值会导致测试频繁失败

Q: 如何保证 benchmark 的公平性？
A:
1. 预热: 跑 3 次丢弃，避免首次编译开销
2. 多次取平均: 跑 100 次取中位数
3. 固定随机种子: torch.manual_seed(41)
4. 控制变量: 相同输入、相同硬件、相同 batch size
```

#### Q5: 如何验证动态 Decode Attention 调度的有效性？

**深度回答**：

```
【问题背景】
静态 split-K 的问题:
- 假设 split-K=4，但请求长度差异大
- 长请求 (10000 tokens) 的 4 个 split 仍然很长
- 短请求 (100 tokens) 的 4 个 split 很短
- CTA 之间负载不均衡

【实验设计】
1. 构造不同长度分布的测试数据:
   - 均匀分布: 所有请求长度相近
   - 长尾分布: 少数请求很长，多数很短
   - 混合分布: 真实场景模拟

2. 测量指标:
   - P99 延迟: 最慢请求的延迟
   - CTA 利用率: 活跃 CTA / 总 CTA
   - 尾部延迟比: P99 / P50

3. 对比方法:
   - 静态 split-K=4
   - 静态 split-K=16
   - 动态 bin-packing (HPC-Ops)

【关键代码】
```python
# hpc/attention.py:515-621
def assign_attention_decode_task(
    num_seq_kvcache: Tensor,
    task_map: Tensor,
    num_head_kv: int,
    mtp: int,
    new_kv_included: bool,
    min_process_len: int = 512,
)
```

Greedy bin-packing 算法:
1. 将所有请求切成统一大小的 KV tile (64 tokens)
2. 按 tile 数量降序排列
3. 贪心地将 tile 分配到负载最小的 CTA
4. 时间复杂度 O(n log n)

【实验结果分析】
典型结果: 最高 2.88x 加速

为什么动态调度更好?
1. 长请求被切分到多个 CTA，并行度提高
2. 短请求合并到同一个 CTA，减少调度开销
3. 负载均衡度提升，尾部延迟降低

【追问准备】
Q: 动态调度的开销是多少？
A:
- Task map 分配: 预分配一次，后续复用
- 任务分配: CPU 端计算，不占用 GPU
- Task map 传递: 每个 task 48 bytes，总开销 < 1ms
- 净收益: 开销 < 1ms，但可以节省 10ms+ 的计算时间

Q: 如何选择 min_process_len？
A:
- 太小: 任务数太多，调度开销大
- 太大: 负载不均衡，失去动态调度的意义
- 经验值: 512 tokens，平衡调度开销和负载均衡
```

### 2.3 回答框架

回答实验验证问题时，建议使用这个框架：

```
1. 实验设计 (Design)
   - 测试什么场景？
   - 对比什么基线？
   - 用什么指标评估？

2. 实验执行 (Execution)
   - 代码怎么实现的？
   - 有什么关键参数？
   - 如何保证公平性？

3. 结果分析 (Analysis)
   - 结果是什么？
   - 为什么有效/无效？
   - 有什么异常点？

4. 追问准备 (Deep Dive)
   - 面试官可能追问什么？
   - 如何解释细节？
```

---

## 3. 第三类：问题定位能力

### 3.1 核心考察点

面试官想看到的：
- 遇到问题时的**排查思路**
- 而不是直接堆砌复杂的解决方案

### 3.2 典型问题与深度回答

#### Q6: 模型上线后推理延迟突然增加 2 倍，怎么排查？

**深度回答**：

```
【排查思路】
分层排查，从外到内:

Level 1: 确认问题范围
- 是所有请求都慢，还是特定请求慢？
- 是突然变慢，还是逐渐变慢？
- 是所有模型都慢，还是特定模型慢？

Level 2: 定位瓶颈环节
- Prefill 慢还是 Decode 慢？
- 是 Attention 慢还是 FFN 慢？
- 是计算慢还是通信慢？

Level 3: 深入具体算子
- 用 profiler 抓取 kernel 时间
- 对比正常和异常时的 kernel 时间
- 找到差异最大的 kernel

【具体排查工具】
1. PyTorch Profiler:
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    model.generate(...)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

2. Nsight Systems:
```bash
nsys profile --trace=cuda,nvtx python3 inference.py
```

3. Compute Sanitizer (HPC-Ops 使用):
```python
# conftest.py
export SANITIZER_CHECK=memcheck,synccheck,racecheck
```

【真实案例】
假设是 Decode Attention 变慢:

1. 检查 batch size 分布:
   - 如果突然涌入大量长请求，会导致 KV Cache 变长
   - 静态 split-K 无法适应，尾部延迟增加

2. 检查任务调度:
   - 参考: hpc/attention.py:582-621
   - 如果 task_map 分配失败，会回退到静态路径
   - 检查 workspace 是否足够

3. 检查内存:
   - KV Cache 内存不足会导致 swap
   - 检查 GPU 显存使用率

【解决方案】
1. 临时方案: 限制 batch size，拒绝新请求
2. 中期方案: 启用动态调度，优化 task_map
3. 长期方案: 升级硬件，优化内存管理

【追问准备】
Q: 如何区分是计算慢还是通信慢？
A:
- 计算慢: kernel 时间增加，但通信时间不变
- 通信慢: AllReduce 时间增加，计算时间不变
- 用 Nsight Systems 可以看到通信和计算的 overlap

Q: 如何快速回滚？
A:
- 保存模型 checkpoint 和配置文件
- 使用 A/B 测试，新版本只分配 10% 流量
- 监控关键指标，异常时自动回滚
```

#### Q7: FP8 Attention 的精度比 BF16 差很多，怎么定位原因？

**深度回答**：

```
【排查思路】
1. 确认是量化问题还是其他问题
2. 定位是哪个量化参数有问题
3. 分析是哪些 token/layer 受影响大

【具体步骤】
Step 1: 对比 FP8 和 BF16 的输出
```python
# tests/utils.py
def calculate_errors(ref_tensor, real_tensor, eps=1e-6, top_k=10):
    abs_error = torch.abs(ref_tensor - real_tensor)
    mae = torch.mean(abs_error).item()
    rel_error = abs_error / (torch.max(torch.abs(ref_tensor), torch.abs(real_tensor)) + eps)
    mre = torch.mean(rel_error).item()
    return {"mean_abs_error": mae, "mean_rel_error": mre, ...}
```

Step 2: 分析误差分布
- 是所有 token 误差都大，还是特定 token？
- 是所有 layer 误差都大，还是深层？
- 是 Q/K 误差大，还是 V 误差大？

Step 3: 检查量化参数
```python
# hpc/attention.py:148-250
# 检查 qscale, kscale, vscale 的分布
print(f"qscale range: {qscale.min()}, {qscale.max()}")
print(f"kscale range: {kscale.min()}, {kscale.max()}")
```

【常见原因】
1. 异常值 (Outlier):
   - 某些 token 的 feature 有极端值
   - FP8 范围 [-448, 448]，超出会被截断
   - 解决: SmoothQuant，对 activation 做 scaling

2. 量化粒度不合适:
   - Per-tensor 量化对分布不均匀的数据损失大
   - 解决: 改用 per-token 或 per-head 量化

3. Scale 计算错误:
   - Scale 应该是 1/max(|x|)，如果计算错误会导致溢出
   - 检查 scale 的计算逻辑

4. 特定层敏感:
   - 浅层 feature 分布更均匀
   - 深层 feature 可能有长尾分布
   - 解决: 敏感层用 BF16

【代码位置】
参考: hpc/attention.py:8-12
```python
class QuantType(Enum):
    QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD = 0
    QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR = 1
```

如果精度不够，尝试从 quant_type=1 切换到 quant_type=0
```

#### Q8: Fused MoE 在某些 batch size 下性能反而下降，怎么分析？

**深度回答**：

```
【排查思路】
性能下降通常是因为:
1. 任务分配不均
2. 内存访问模式差
3. 并行度不足

【具体分析】
Step 1: 对比不同 batch size 的性能曲线
- 正常情况: 性能随 batch size 增加而提升
- 异常情况: 某个 batch size 性能突然下降

Step 2: 分析 kernel 时间 breakdown
```python
# 用 profiler 分解每个阶段的时间
count_and_gather_time = ...
gate_up_gemm_time = ...
activation_time = ...
down_gemm_time = ...
reduce_time = ...
```

Step 3: 检查任务分配
```python
# hpc/fuse_moe.py:8-85
# 检查每个 expert 的 token 数量分布
print(f"seqlens: {seqlens}")
print(f"max seqlens: {seqlens.max()}, min seqlens: {seqlens.min()}")
```

【常见原因】
1. 负载不均衡:
   - 某些 expert 被过多 token 选择
   - 其他 expert 空闲
   - 解决: 调整 router 的 capacity factor

2. 小 batch 时 kernel launch 开销占比高:
   - 小 batch 时计算量小，但 kernel launch 开销固定
   - Fused MoE 的融合收益在小 batch 时不明显
   - 解决: 小 batch 时回退到非融合路径

3. TMA 配置开销:
   - 参考: src/fuse_moe/fuse_moe.cu vs src/fuse_moe/cp_async/fuse_moe.cu
   - 小 intermediate_size 时 TMA 开销大于收益
   - 解决: intermediate_size <= 512 时用 cp.async 路径

【代码位置】
```python
# hpc/fuse_moe.py:133
_CP_ASYNC_N_TP_MAX = 512  # 阈值: intermediate_size <= 512 用 cp.async
```

【追问准备】
Q: 如何选择 TMA 和 cp.async 的阈值？
A:
- 实验方法: 扫描不同 intermediate_size，对比两条路径的性能
- 阈值选择: 性能交叉点，通常是 512
- 动态调整: 可以根据实际 workload 动态切换
```

### 3.3 回答框架

回答问题定位问题时，建议使用这个框架：

```
1. 确认问题 (Confirm)
   - 问题的具体表现是什么？
   - 影响范围有多大？
   - 是突然出现还是逐渐恶化？

2. 分层排查 (Layer by Layer)
   - 从外到内，逐层缩小范围
   - 使用合适的工具 (profiler, sanitizer)
   - 对比正常和异常情况

3. 定位根因 (Root Cause)
   - 是什么导致了这个问题？
   - 为什么之前没有出现？
   - 有什么触发条件？

4. 解决方案 (Solution)
   - 临时方案: 快速止血
   - 中期方案: 解决根因
   - 长期方案: 预防再次发生
```

---

## 4. 第四类：工程落地能力

### 4.1 核心考察点

面试官想看到的：
- **理论可行 != 工程可行**
- 实际部署时遇到的问题和解决方案
- 系统稳定性、监控、回滚等工程细节

### 4.2 典型问题与深度回答

#### Q9: 算子库怎么保证生产环境的稳定性？

**深度回答**：

```
【测试体系】
1. 单元测试: 26 个测试文件，覆盖所有算子
```python
# tests/test_fuse_moe_pertensor.py
@pytest.mark.parametrize("num_seq", [128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [512])
...
def test_fuse_moe_pertensor_fp8(...):
    my = hpc.fuse_moe_pertensor_fp8(...)
    gt = naive_fuse_moe_pertensor_fp8(...)
    assert allclose(gt, my, rtol=0.08, atol=0.1)
```

2. 参数化测试: 覆盖不同配置
   - 不同 batch size
   - 不同 sequence length
   - 不同 expert 数量
   - 不同 EP 配置

3. 精度验证: 对比 naive 实现
   - MAE (Mean Absolute Error)
   - MRE (Mean Relative Error)
   - Top-K 误差分析

【Sanitizer 集成】
参考: conftest.py

```python
# 通过环境变量开启 compute-sanitizer
export SANITIZER_CHECK=memcheck,synccheck,racecheck

# 自动对每个 hpc 函数进行检查
# memcheck: 内存越界
# synccheck: 同步错误
# racecheck: 竞态条件
```

【发布流程】
1. 代码 Review: 至少两人 review
2. CI/CD: 自动跑所有测试
3. 灰度发布: 新版本先分配 10% 流量
4. 监控告警: 延迟、错误率、内存使用
5. 快速回滚: 保留旧版本，异常时秒级切换

【监控指标】
1. 性能指标:
   - P50/P99 延迟
   - 吞吐 (tokens/s)
   - GPU 利用率

2. 精度指标:
   - 输出与 baseline 的误差
   - 特定 token 的 logit 分布

3. 资源指标:
   - GPU 显存使用
   - CPU 内存使用
   - 网络带宽

【追问准备】
Q: 如何处理 CUDA kernel 的不确定性？
A:
- 固定随机种子: torch.manual_seed(41)
- 使用 deterministic 算法: torch.use_deterministic_algorithms(True)
- 但某些优化 (如 atomicAdd) 本身不确定，需要在精度允许范围内

Q: 如何处理不同 GPU 型号的兼容性？
A:
- 编译时指定架构: CUDA_ARCHITECTURES "90a"
- 运行时检查: torch.cuda.get_device_capability()
- 不支持的架构回退到 PyTorch 原生实现
```

#### Q10: 如何处理大规模部署时的内存管理？

**深度回答**：

```
【内存管理挑战】
1. KV Cache 内存随请求动态变化
2. 不同请求的 KV Cache 长度差异大
3. 需要在性能和内存之间平衡

【解决方案】
1. Paged KV Cache:
```python
# hpc/attention.py:70-145
block_ids: Tensor  # [num_batch, max_blocks] - page table
kcache: Tensor     # [num_blocks, block_size, num_head_kv, dim]
```
- 按需分配 block，用完即释放
- 支持跨请求共享 (如 system prompt)

2. 内存池管理:
- 预分配 block pool
- 避免频繁的 cudaMalloc/cudaFree
- 使用 free list 管理空闲 block

3. 动态调整:
- 根据 GPU 显存使用率调整 batch size
- 显存不足时拒绝新请求或降级服务

【代码实现】
```python
# hpc/attention.py:515-579
def get_attention_decode_task_workspace(
    max_num_batch: int, max_seqlen: int, num_head_kv: int, min_process_len: int = 512
):
    # 预分配 workspace，避免运行时分配
    workspace_byte_size = ...
    workspace = torch.zeros(workspace_byte_size, dtype=torch.int8, device="cuda")
    return workspace
```

【工程细节】
1. 预分配策略:
   - 根据预期的最大 batch size 预分配
   - 避免运行时 cudaMalloc 的开销和碎片

2. 内存复用:
   - Task map 复用，避免重复分配
   - Workspace 跨请求复用

3. OOM 处理:
   - 捕获 CUDA OOM 异常
   - 降级服务 (减少 batch size)
   - 或拒绝新请求

【追问准备】
Q: 如何估算 KV Cache 的内存需求？
A:
```
KV Cache 内存 = 2 * num_layers * num_heads * head_dim * seq_len * batch_size * dtype_size
```
例如: 32 layers, 32 heads, 128 dim, BF16, batch=64, seq=4096
= 2 * 32 * 32 * 128 * 4096 * 64 * 2 bytes = 16 GB

Q: Paged Attention 的 block_size 如何选择？
A:
- 太小: page table 管理开销大
- 太大: 内部碎片多，内存浪费
- 经验值: 16 tokens，在管理开销和内存效率之间平衡
```

### 4.3 回答框架

回答工程落地问题时，建议使用这个框架：

```
1. 工程挑战 (Challenge)
   - 理论和实际的差距在哪里？
   - 有什么工程约束？

2. 解决方案 (Solution)
   - 如何设计系统架构？
   - 使用了什么工程技巧？

3. 稳定性保障 (Stability)
   - 如何测试？
   - 如何监控？
   - 如何回滚？

4. 实际效果 (Result)
   - 在生产环境的表现如何？
   - 有什么经验教训？
```

---

## 5. 第五类：业务与实际场景的理解

### 5.1 核心考察点

面试官想看到的：
- 理解**技术方案的业务价值**
- 知道**用户真正关心什么**
- 能在**资源有限时做优先级排序**

### 5.2 典型问题与深度回答

#### Q11: HPC-Ops 的优化对业务有什么实际价值？

**深度回答**：

```
【业务价值分析】
1. 直接成本节省:
   - 推理延迟降低 -> 同样 QPS 需要更少的 GPU
   - 假设延迟从 100ms 降到 50ms，吞吐翻倍
   - GPU 成本节省 50%

2. 用户体验提升:
   - 延迟降低 -> 用户等待时间减少
   - 吞吐提升 -> 高峰期不丢请求
   - 首 token 延迟 (TTFT) 降低 -> 用户感知更快

3. 产品能力增强:
   - 支持更长上下文 (128K+)
   - 支持更大 batch size -> 更高并发
   - 支持更大模型 (如 DeepSeek-V3)

【不同场景的需求差异】
1. 在线对话 (Chat):
   - 用户最关心: 首 token 延迟 (TTFT)
   - 优化重点: Prefill 性能
   - 典型需求: TTFT < 500ms

2. 批处理 (Batch Processing):
   - 用户最关心: 总成本
   - 优化重点: 吞吐 (tokens/s)
   - 典型需求: 尽可能高的吞吐

3. 长文档理解:
   - 用户最关心: 能否处理长文档
   - 优化重点: 内存效率
   - 典型需求: 支持 128K+ 上下文

4. 实时交互:
   - 用户最关心: 响应流畅度
   - 优化重点: Decode 延迟
   - 典型需求: 每 token < 20ms

【HPC-Ops 的价值定位】
1. Attention 优化:
   - 动态调度降低尾部延迟
   - 对在线对话场景价值最大

2. Fused MoE:
   - 降低 MoE 模型的推理成本
   - 对大模型 (DeepSeek-V3) 部署价值大

3. FP8 量化:
   - 内存减半，支持更大 batch
   - 对高并发场景价值大

【追问准备】
Q: 如果资源有限，应该优先优化哪个部分？
A:
分析方法:
1. Profiling 找到瓶颈: 哪个算子占时间最多？
2. 评估优化收益: 优化后能提升多少？
3. 评估实现成本: 需要多少人力/时间？

优先级排序:
1. 如果是 Decode-bound: 优先优化 Decode Attention
2. 如果是 Memory-bound: 优先优化 KV Cache 管理
3. 如果是 Compute-bound: 优先优化 GEMM/MoE

典型优先级:
- Decode Attention 动态调度 (收益高，成本中)
- FP8 量化 (收益高，成本低)
- Fused MoE (收益中，成本高)
```

#### Q12: 这个方案适合什么规模的模型和场景？

**深度回答**：

```
【适用场景分析】
1. 模型规模:
   - 适合: 7B+ 的模型
   - 不适合: 小模型 (1-3B)，优化收益不明显
   
2. 部署规模:
   - 适合: 单机多卡 (TP=8) 或多机多卡 (EP=8)
   - 不适合: 单卡部署，通信开销无法摊薄

3. 业务场景:
   - 适合: 高并发在线服务
   - 不适合: 低并发离线任务

【具体模型适配】
参考: benchmark/fused_moe/benchmark_fuse_moe.py

```python
MODEL_PRESETS = {
    "qwen3-235b":  (128, 8, 4096, 1536),  # 适合
    "hunyuan-v3":  (192, 8, 4096, 1536),  # 适合
    "deepseek-v3": (256, 8, 7168, 2048),  # 最适合
}
```

为什么 DeepSeek-V3 最适合?
- 256 experts，MoE 优化收益大
- 7168 hidden_size，GEMM 优化收益大
- 支持 TP=8 EP=8，并行优化收益大

【不适合的场景】
1. 小模型 (1-3B):
   - 计算量小，优化收益不明显
   - 内存充足，不需要 FP8 量化

2. 单卡部署:
   - 没有通信开销，不需要 AllReduce 优化
   - MoE 的 EP 优化无法使用

3. 低并发场景:
   - Batch size 小，kernel 融合收益小
   - 可以用简单的 PyTorch 实现

【成本收益分析】
收益:
- GPU 成本节省 30-50%
- 延迟降低 20-40%
- 支持更长上下文

成本:
- 开发成本: 需要 CUDA 专家
- 维护成本: 需要持续优化
- 部署成本: 需要适配不同硬件

【追问准备】
Q: 如果要支持新的模型架构，需要做什么？
A:
1. 分析新架构的算子需求
2. 评估现有算子的复用性
3. 开发新算子 (如果需要)
4. 测试和优化

Q: 如何评估优化 ROI？
A:
```
ROI = (优化收益 - 优化成本) / 优化成本

优化收益 = GPU 节省 + 延迟降低带来的业务价值
优化成本 = 开发人力 + 维护成本 + 硬件成本
```
```

### 5.3 回答框架

回答业务理解问题时，建议使用这个框架：

```
1. 业务价值 (Value)
   - 对用户有什么价值？
   - 对公司有什么价值？

2. 场景适配 (Scenario)
   - 适合什么场景？
   - 不适合什么场景？

3. 成本收益 (Cost-Benefit)
   - 收益是什么？
   - 成本是什么？
   - ROI 如何？

4. 优先级排序 (Priority)
   - 如果资源有限，先做什么？
   - 如何做决策？
```

---

## 6. 综合模拟面试问答

### 6.1 完整面试流程模拟

**面试官**: 请介绍一下你参与的 HPC-Ops 项目。

**候选人**:
> HPC-Ops 是腾讯混元团队开发的高性能 LLM 推理算子库，我在其中学习了 Attention、MoE、量化等核心优化技术。
> 
> 项目的核心价值是降低 LLM 推理成本。以 Fused MoE 为例，我们将路由、Gate-Up GEMM、激活、Down GEMM、归约这 5 个步骤融合成一个流水线，减少了 kernel launch 开销和中间数据的内存访问。在 DeepSeek-V3 上实测可以达到 1.6x 的加速。
> 
> 我主要学习了三个方面的内容：
> 1. **Attention 优化**: 包括 KV Cache 管理、FP8 量化、动态任务调度
> 2. **MoE 架构**: 包括 Fused MoE 流程、EP/TP 并行策略
> 3. **工程实践**: 包括测试体系、性能分析、生产部署

---

**面试官**: 你提到 FP8 量化，能详细讲讲 FP8 量化的精度风险吗？

**候选人**:
> FP8 量化的核心风险是**动态范围受限**和**误差累积**。
> 
> FP8 E4M3 的范围是 [-448, 448]，如果 feature 有异常值，会被截断。更重要的是，在 Attention 计算中，Q 和 K 的量化误差会在 Q@K^T 中累积，然后被 softmax 放大。
> 
> 我们在代码中使用了两种量化策略来缓解：
> - **quant_type=1**: Q per-token per-head + K/V per-tensor，这是平衡精度和开销的默认选择
> - **quant_type=0**: Q/K/V 都用 per-token per-head，精度更高但开销更大
> 
> 验证精度时，我们对比 naive PyTorch 实现，用 MAE 和 MRE 评估，设定 rtol=0.08, atol=0.1 的阈值。

---

**面试官**: 你们怎么验证动态 Decode Attention 调度的有效性？

**候选人**:
> 我们设计了三类测试场景：
> 1. **均匀分布**: 所有请求长度相近，验证基线性能
> 2. **长尾分布**: 少数请求很长，多数很短，验证负载均衡
> 3. **混合分布**: 模拟真实场景
> 
> 对比基线是静态 split-K=4 和 split-K=16。关键指标是 P99 延迟和 CTA 利用率。
> 
> 实验结果显示，在长尾分布下，动态调度可以达到 2.88x 的加速。原因是静态 split-K 无法适应长度差异，导致长请求的 CTA 一直忙碌，短请求的 CTA 一直空闲。动态调度通过 greedy bin-packing 将任务均匀分配，提升了 CTA 利用率。
> 
> 动态调度本身的开销很小：task map 预分配，任务分配在 CPU 端完成，每个 task 只有 48 bytes，总开销 < 1ms。

---

**面试官**: 如果上线后推理延迟突然增加 2 倍，你怎么排查？

**候选人**:
> 我会分层排查：
> 
> **第一步：确认问题范围**
> - 是所有请求都慢，还是特定请求慢？
> - 是 Prefill 慢还是 Decode 慢？
> 
> **第二步：用 Profiler 定位**
> - 用 PyTorch Profiler 抓取 kernel 时间
> - 对比正常和异常时的 kernel 时间
> - 找到差异最大的 kernel
> 
> **第三步：深入分析**
> - 如果是 Decode Attention 变慢，检查 batch size 分布和 task_map 分配
> - 如果是 MoE 变慢，检查 expert 的负载分布
> - 如果是 AllReduce 变慢，检查网络状况
> 
> **第四步：解决**
> - 临时方案：限制 batch size，拒绝新请求
> - 中期方案：调整调度策略，优化 task_map
> - 长期方案：升级硬件或优化算法

---

**面试官**: 这个方案适合什么场景？如果资源有限，应该优先优化哪个部分？

**候选人**:
> **适用场景**：
> - 7B+ 的大模型，小模型优化收益不明显
> - 单机多卡或多机多卡部署，单卡无法发挥并行优势
> - 高并发在线服务，低并发场景收益小
> 
> **不适合的场景**：
> - 1-3B 的小模型，计算量小，优化收益有限
> - 单卡部署，没有通信开销需要优化
> - 离线批处理，对延迟不敏感
> 
> **优先级排序**（如果资源有限）：
> 1. **Decode Attention 动态调度** - 收益高，成本中等，对在线场景价值最大
> 2. **FP8 量化** - 收益高，成本低，可以快速落地
> 3. **Fused MoE** - 收益中等，成本高，适合大模型
> 
> 决策依据：
> - 用 Profiling 找到瓶颈在哪里
> - 评估优化收益和实现成本
> - 优先优化瓶颈环节

---

### 6.2 面试注意事项

1. **不要背概念，要讲理解**
   - ❌ "Paged Attention 是一种内存管理技术"
   - ✅ "Paged Attention 解决了三个问题：内存预分配浪费、碎片化、并发动态批处理困难"

2. **不要只讲结果，要讲过程**
   - ❌ "我们优化后性能提升了 2 倍"
   - ✅ "我们通过 Profiling 发现瓶颈在 Decode Attention，然后用动态调度优化了负载均衡"

3. **不要回避问题，要展示思考**
   - ❌ "这个我不太清楚"
   - ✅ "这个问题我没遇到过，但我的思路是..."

4. **不要过度承诺，要诚实评估**
   - ❌ "这个方案可以解决所有问题"
   - ✅ "这个方案在高并发场景下效果好，但低并发场景收益有限"

---

## 附录: 关键代码索引

| 功能 | 文件位置 | 关键行号 |
|------|----------|----------|
| Paged KV Cache | `hpc/attention.py` | 70-145 |
| FP8 量化策略 | `hpc/attention.py` | 8-12, 148-250 |
| 动态调度 | `hpc/attention.py` | 515-621 |
| Fused MoE | `hpc/fuse_moe.py` | 136-166 |
| BF16xFP32 GEMM | `hpc/gemm.py` | 全文 |
| AllReduce+RMSNorm | `hpc/allreduce.py` | 全文 |
| Fused Sampler | `hpc/sampler.py` | 42-182 |
| 测试框架 | `tests/utils.py` | 全文 |
| 性能基准 | `benchmark/fused_moe/benchmark_fuse_moe.py` | 全文 |
| Sanitizer 集成 | `conftest.py` | 全文 |

---

*本文档基于 HPC-Ops 项目深度分析，针对 LLM 算法实习面试的五类核心能力准备。建议结合代码阅读和实际实验加深理解。*

*最后更新: 2026 年 6 月*
