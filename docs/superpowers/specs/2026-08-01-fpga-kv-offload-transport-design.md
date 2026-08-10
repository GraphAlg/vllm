# FPGA KV Cache Offload — 传输层架构设计

> 状态:草稿(待评审)
> 日期:2026-08-01
> 范围:重新设计 `vllm/v1/kv_offload/fpga/` 的传输层(数据通路),支持多厂商硬件,修复已验证的架构问题。

---

## 1. 背景与现状

### 1.1 目标

基于 vLLM 的 `kv_offload` 框架,将 KV cache 卸载到 FPGA 扩展内存,以扩展推理可用的 KV 容量。本文档重新设计其中的**传输层**(GPU↔FPGA 数据通路),解决现有实现暴露的架构问题,并使系统能在不同硬件条件下工作。

### 1.2 现状架构

现有实现已经完整接入 vLLM 上游的 kv_offload 框架,分层如下:

```
OffloadingSpec (FPGAOffloadingSpec)     容量/块大小计算,注册于 OffloadingSpecFactory
  └─ OffloadingManager (FPGAOffloadingManager)   块分配/逐出跟踪,委托 CPUOffloadingManager(LRU/ARC)
      └─ OffloadingWorker (FPGAOffloadingWorker)  异步传输调度,streams + CUDA event 完成跟踪
          └─ BlockTransferEngine                  后台线程串行化传输,块/层逐块拷贝
              └─ DMABackend (抽象)                vendor 无关的物理传输接口
                  ├─ GPUDirectP2PBackend           GPU DMA 直写 FPGA BAR(cuMemHostRegister IOMEMORY)
                  ├─ XDMABounceBufferBackend       Xilinx XDMA + host 反弹缓冲(兼容回退)
                  └─ IntelFPGAP2PBackend           Intel FPGA 驱动(骨架,未实现)
```

接入方式:`--kv-transfer-config` JSON 指定 `kv_connector=OffloadingConnector` + `spec_name=FPGAOffloadingSpec`。后端由 `VLLM_FPGA_BACKEND` 环境变量选择。

### 1.3 已验证的架构问题

以下问题在本轮调试中被实际复现并定位(均有日志/脚本证据):

**P1. 后端依赖不可靠的 CUDA 原语,且失败处理是"硬失败"**
`GPUDirectP2PBackend` 完全依赖 `cuMemHostRegister(IOMEMORY)` + `cuMemcpyAsync` D→D。`IOMEMORY` 是 NVIDIA 的 niche 特性,对第三方 FPGA BAR 属非官方用法。实测:
- 某些环境注册阶段直接失败(`CUDA_ERROR_*`);
- 即使注册成功,**注册成功 ≠ aperture 是活的** —— `verify_kv_offload.py` 证明 store 通路完整执行,但 FPGA 读回全 `0xff`(`diag_dma_data.py` Phase C 纯 host 写读也是 `0xff`)。
  - **硬件状态是动态的**:同一 `diag_dma_data.py`,07-31 全 FAIL(0xff,当时 CvP disabled / 无 bitstream / BAR 解码未启用),08-03 全 PASS(FPGA 已配置)。这正是 probe 必须**每次启动实时回环验证**、而不是缓存假设的原因。
- 后端选择失败即启动崩溃,无回退。

**P2. CUDA 上下文泄漏,破坏 torch/Triton 状态**
`GPUDirectP2PBackend._map_and_register_bar` 在主线程调用 `cuCtxSetCurrent` 后不恢复 → torch 缓存的 current device 与实际 CUDA context 脱钩 → 引擎 warmup 的 Triton kernel 启动失败(`Pointer argument cannot be accessed from Triton`)。已修复(保存/恢复),但机制必须固化为接口契约。

**P3. device 不匹配,跨上下文拷贝必失败**
后端从 `VLLM_FPGA_DEVICE_ID`(默认 0)取 device,与 vLLM 实际跑模型的 device 可能不一致 → BAR 注册在错误的 context,`cuMemcpyAsync` 跨上下文。已修复(默认跟随 `torch.cuda.current_device()` + 告警),同样需固化为契约。

**P4. 传输通路决策靠鸭子类型**
`BlockTransferEngine` 用 `hasattr(backend, "bounce_buffer")` 判断走 bounce 还是直写([copy_backend.py](vllm/v1/kv_offload/fpga/copy_backend.py#L69-L73))。后端没有能力声明机制,新硬件接不进来。

**P5. "异步"引擎被同步后端拖垮**
P2P 后端每次拷贝后 `cuStreamSynchronize`([gpu_direct_p2p.py](vllm/v1/kv_offload/fpga/backends/gpu_direct_p2p.py#L349-L352)),bounce 路径也同步等待。逐块逐层串行拷贝 + 单后台线程,带宽受限于串行同步。

**P6. 硬件现实:厂商差异未被抽象**
实测板卡为 Altera/Intel FPGA(PCI `0000:88:00.0`,Region 2 = 4GB prefetchable),驱动体系为 OPAE/DFL,而非 Xilinx XDMA。硬编码的 `gpu_direct_p2p`(GPU 直写 BAR)在该板上还受制于 FPGA 未编程的硬件前置条件(CvP disabled → 无 bitstream → aperture 读 `0xff`)。架构必须**按厂商/驱动/回环验证选择通路**,而不是猜。

### 1.4 需求

- 支持不同硬件条件下的 P2P DMA 传输方式:如 Xilinx FPGA 用 XDMA 驱动、Intel FPGA 用其自有驱动、GPU 直写 BAR 路径作为可用时的高性能选项。
- 传输通路不可用时优雅降级,而不是启动即崩。
- 保持 vLLM kv_offload 框架集成不变(spec / manager / connector)。

---

## 2. 设计目标与边界

### 2.1 目标

1. **厂商无关传输抽象**:一套 `DMABackend` 接口,覆盖 Xilinx XDMA / Intel OPAE / GPU-direct 等物理通路。
2. **能力声明 + 真实回环探测**:后端自称可用不算数,必须通过"写已知数据 → 读回 → 对比"的验证(吸取 P1 教训)。
3. **自动选择 + 优雅回退**:启动时按 vendor/驱动/回环结果探测,失败自动降级到已知可用的基础通路。
4. **上下文/device 契约**:任何后端不得泄漏当前线程 CUDA 上下文;传输 device 必须与模型一致(固化 P2/P3 修复)。
5. **真异步**:后端内部不得用同步等待破坏引擎的异步模型。

### 2.2 非目标

- **不实现** Intel OPAE/DFL 驱动的底层细节 —— 那是 Intel 驱动侧的事;本设计只定义接入契约,驱动调用封装在 `IntelFPGAP2PBackend` 内部。
- **不解决** FPGA bitstream 加载/编程 —— 这是硬件前置条件,在 vLLM 之外;probe 只能检测"当前不可用"并回退,不能代替配置。
- **不做** 完整 plugin 框架 —— 现阶段只有三种后端形态,能力模型 + 探测已足够(YAGNI)。
- **不改变** vLLM kv_offload 框架的集成层(`OffloadingSpec`/`OffloadingManager`/`OffloadingConnector`)。

---

## 3. 架构总览

分层结构保持不变,改动集中在**传输层**:

```
(不变) OffloadingSpec → OffloadingManager → OffloadingWorker
                                                      │
(重设计)                                            TransferEngine
                                                      │  按后端能力驱动数据流
                                                      ▼
                                            DMABackend (抽象)
                                            ├── 能力声明 BackendCapabilities
                                            ├── 类方法 probe() → ProbeResult (含回环验证)
                                            └── 上下文/device 契约
                                                      │
                                  ┌───────────────────┼───────────────────┐
                                  ▼                   ▼                   ▼
                      GPUDirectP2PBackend   IntelFPGAP2PBackend   XDMABounceBufferBackend
                      (重构:能力+probe)     (实现:OPAE/DFL)      (基础保底)
```

设计原则:

- **Manager 不变**:块跟踪、LRU/ARC 逐出逻辑与 medium 无关,已证明正确。
- **Worker 微调**:后端创建从"环境变量选一个"改为"探测链选第一个可用"。
- **Engine 重设计**:数据流由 `capabilities` 驱动,不再鸭子类型;同步后端由后台线程承载异步。
- **Backend 增强**:能力声明 + 探测协议 + 上下文契约。

---

## 4. 核心设计

### 4.1 DMABackend 能力模型

新增 `BackendCapabilities` 数据类,后端在构造时声明自己的能力。引擎据此选择数据流,不再猜。

```python
@dataclass(frozen=True)
class BackendCapabilities:
    # write()/read() 是否在 CUDA stream 上异步完成(True → 引擎用 CUDA event 跟踪完成,
    # 不额外同步;False → 调用本身阻塞,由引擎后台线程承载)
    stream_async: bool = False

    # 是否需要 host 反弹缓冲中转(True → GPU↔host↔FPGA 两跳;False → 单跳直连)
    needs_bounce: bool = False

    # 支持的最大单次传输字节数(供引擎做块合并)
    max_transfer_bytes: int = 0  # 0 = 不限制

    # 地址对齐要求(字节)
    alignment: int = 1
```

| 后端 | stream_async | needs_bounce | 说明 |
|---|---|---|---|
| GPUDirectP2P | True | False | cuMemcpyAsync 在 CUDA stream 上,当前实现内部同步 → 需重构为真异步(P5) |
| XDMABounce | False | True | XDMA 无 CUDA 参与,调用阻塞,引擎后台线程承载 |
| IntelFPGA(OPAE) | False(默认) | True(默认) | 接入后按实际驱动能力声明 |

### 4.2 探测协议

每个后端实现**类方法** `probe()`,返回结构化结果。**Probe 必须包含真实回环验证**,不能只看资源存在。

```python
@dataclass(frozen=True)
class ProbeResult:
    available: bool
    reason: str | None = None        # unavailable 时的原因
    # available=True 时可附加:探测到的 device/地址/版本信息

class DMABackend(ABC):
    @classmethod
    @abstractmethod
    def probe(cls, ctx: "ProbeContext") -> ProbeResult: ...
```

`ProbeContext` 提供 probe 所需环境:`model_device`(torch 当前设备)、`fpga_capacity_bytes`、`xdma_prefix`、环境变量。

各后端 probe 步骤:

**GPUDirectP2PBackend.probe()**
1. 环境:确认 CUDA driver 可用、FPGA PCI 设备存在(`/sys/bus/pci/devices/<bdf>/resource<bar>` 可访问,或从 vendor ID 自动发现 BDF/BAR)。
2. 地址窗口有效性:mmap BAR 后读几个 word,断言非 `0xff`(0xff = 设备不解码地址)。
3. **回环验证**:host 写已知 pattern → host 读回对比;再用 `cuMemHostRegister(IOMEMORY)` + `cuMemcpyAsync` 写→读回对比(即 `verify_bar.c` 的逻辑)。
4. 上下文卫生:整个 probe 过程保存/恢复当前 CUDA 上下文(契约见 4.5)。

**XDMABounceBufferBackend.probe()**
1. 环境:`/dev/xdma*_h2c_0` / `_c2h_0` 存在(或 `VLLM_FPGA_MOCK=1`)。
2. 可选回环:通过 XDMA 写/读一个已知 pattern(存在 mock 时跳过)。

**IntelFPGAP2PBackend.probe()**
1. 环境:检测 OPAE/DFL(`libopae-c.so`、`/dev/dfl-*`、`fpgaOpen` 可调用)或 vendor 库。
2. 回环:驱动级写读验证。

### 4.3 后端选择与回退链

在 `FPGAOffloadingWorker._setup` 中,后端创建改为**按优先级探测**:

```python
_BACKEND_PRIORITY = [
    "gpu_direct_p2p",   # 单跳直连,性能最佳,但受硬件前置条件约束
    "intel_fpga",       # Intel 驱动通路(OPAE/DFL)
    "xdma_bounce",      # 基础保底:兼容回退
]
```

规则:
1. 依次对 `_BACKEND_PRIORITY` 调用 `probe()`,取第一个 `available=True` 的。
2. `VLLM_FPGA_BACKEND` 显式指定时:**优先尝试指定后端**,若其 probe 失败 → 打告警,继续按优先级探测(而非崩溃)。
3. probe 全部失败 → 明确报错并列出每个后端的失败原因(此时是配置错误,应快速失败,而不是带着死通路运行)。
4. probe 结果与选择理由打 INFO 日志,便于排障。

### 4.4 TransferEngine 能力驱动数据流

`BlockTransferEngine` 重设计:

- **移除鸭子类型**:`hasattr(backend, "bounce_buffer")` 删除,改用 `capabilities.needs_bounce`。
- **数据流分支**:
  - `needs_bounce=False`(直连):store = `backend.write(gpu_ptr, fpga_addr, size, stream)`;load = `backend.read(fpga_addr, gpu_ptr, size, stream)`。单跳。
  - `needs_bounce=True`(两跳):store = GPU→bounce(stream)→ `backend.write`;load = `backend.read` → bounce→GPU(stream)。
- **同步后端承载**:`stream_async=False` 的后端调用本身阻塞,由引擎后台线程串行化,天然异步化(现状已如此,保留)。
- **真异步直连**:`stream_async=True` 的后端,`write/read` 只下发到 stream,引擎用 CUDA event 跟踪完成(现状的 `_poll_events`/`wait_for_copy` 机制保留),**去掉后端内部的 `cuStreamSynchronize`**(P5)。
- **块合并(可选增强)**:在 `max_transfer_bytes` 内把多个 block 合并成一次传输,减少 PCIe 事务开销。

### 4.5 上下文/device 契约

写入 `DMABackend` 接口文档,作为所有后端实现的强制约束:

1. **不泄漏当前线程 CUDA 上下文**:任何需要在别的 context 下执行的操作(如 `cuMemHostRegister`),必须先 `cuCtxGetCurrent` 保存,操作后 `cuCtxSetCurrent` 恢复(参考已修复的 [gpu_direct_p2p.py](vllm/v1/kv_offload/fpga/backends/gpu_direct_p2p.py#L261-L294) 模式)。禁止在 torch 活跃的主线程上永久切换 context。
2. **device 必须与模型一致**:后端使用的 CUDA device 由 worker 传入(`model_device`),不从硬编码或独立环境变量取。若后端确需不同 device,必须显式说明并承担跨上下文风险。
3. **probe 同样遵守**:探测过程也要保存/恢复上下文,probe 不得污染引擎状态。

### 4.6 异步模型总结

| 层 | 异步机制 |
|---|---|
| Worker | store/load 提交即返回;`_poll_events` 用 CUDA event 查询完成;`wait()` 按需同步 |
| Engine | 后台线程串行化传输;`wait_for_copy` 用 threading.Event + CUDA event 完成信号 |
| Backend (stream_async) | 只下发到 stream,不内部同步 |
| Backend (blocking) | 调用阻塞,由引擎后台线程承载 |

---

## 5. 数据流

KV cache offload 横跨**调度侧**(engine core)与 **worker 侧**(model runner),两侧各有一个 connector 实例,通过 `OffloadingConnectorMetadata` 传递 job。本节给出端到端生命周期;物理传输层(能力驱动分支)在 5.3。

### 5.1 调度侧:何时卸载(每步执行)

engine 每步调度调用 `connector.build_connector_meta(scheduler_output)`([sched/scheduler.py:1134](vllm/v1/core/sched/scheduler.py#L1134)),在 `OffloadingConnectorScheduler` 内([offloading/scheduler.py](vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py)):

```
_update_req_states()     # 更新每个请求的 offload keys(block_hash + group_idx 序列)
_build_load_jobs()       # 前缀命中 FPGA 但不在 GPU 的块 → load job
_build_store_jobs()      # 可卸载块 → store job
```

**Store 触发条件**([scheduler.py:844](vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L844) `_build_store_jobs`):

```
num_offloadable_tokens = min(已计算token数, 请求token数)
                          # offload_prompt_only=True(默认)时再钳制到 prompt tokens
num_blocks = num_offloadable_tokens // offloaded_block_size
if num_blocks <= next_stored_block_idx: 跳过      # 不足一块 → 不触发
manager.prepare_store(keys) → 分配 FPGA 块,LRU/ARC 逐出 → (store_spec, evicted_keys)
```

- store 按**已计算 token 渐进触发**,不依赖 GPU eviction
- 短 prompt(不足一块)不触发 —— 实测中"无 Store"的原因;长 prompt 实测 `Store: 125 GPU blocks → FPGA`
- `offload_prompt_only=False` 时 decode 阶段的块也参与卸载

**Load 触发条件**(`_build_load_jobs`):请求的哈希前缀在 manager `lookup()` 命中(`HIT`)且该块不在 GPU cache → 生成 load job。

### 5.2 worker 侧:job → 实际传输(每步开头)

metadata 送到 worker 侧,`OffloadingConnectorWorker` 提交([offloading/worker.py:281](vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py#L281)):

```
start_kv_transfers(metadata)
  ├─ 提交积压 store:worker.submit_store(job_id, gpu_blocks, fpga_blocks)
  └─ 每个 load job: worker.submit_load(job_id, fpga_blocks, gpu_blocks)

get_finished()
  ├─ prepare_store_kv(metadata)  # 新 store job 推迟到下一步开头提交,避免延迟 token 生成
  └─ worker.get_finished()       # 收集完成的 TransferResult → 调度侧 complete_store/complete_load
```

设计要点:**store 故意推迟到下一步开头**([worker.py:294](vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py#L294) 注释),让 offload 不挡当步的 token 生成。

### 5.3 物理传输层(能力驱动,本设计的核心)

`FPGAOffloadingWorker.submit_store/submit_load` → `BlockTransferEngine.launch_copy` → 后台线程 `_copy_loop` → `_do_store/_do_load`,按 `BackendCapabilities.needs_bounce` 分支:

**Store(GPU → FPGA)**

```
Worker.submit_store(gpu_blocks, fpga_blocks)
  └─ 记录 store_compute_done(event: GPU compute 完成)
      └─ Engine.launch_copy(is_store=True, wait_event=store_compute_done)
          └─ 后台线程 _do_store:
              needs_bounce=False(直连):
                gpu_src = gpu_tensor[bid].contiguous()
                backend.write(gpu_src.data_ptr(), fpga_addr, size, stream=store_stream)
              needs_bounce=True(两跳):
                bounce[:size].copy_(gpu_src, non_blocking=True)   # on store_stream
                backend.write(bounce.data_ptr(), fpga_addr, size)
              └─ record event on store_stream → 完成信号
```

**Load(FPGA → GPU)**

```
Worker.submit_load(fpga_blocks, gpu_blocks)
  └─ Engine.launch_copy(is_store=False)
      └─ 后台线程 _do_load:
          needs_bounce=False:
            backend.read(fpga_addr, gpu_tensor[bid].data_ptr(), size, stream=load_stream)
          needs_bounce=True:
            backend.read(fpga_addr, bounce.data_ptr(), size)
            gpu_tensor[bid].copy_(bounce[:size], non_blocking=True)  # on load_stream
          └─ record event on load_stream → 完成信号
```

### 5.4 生命周期中的 block 状态

```
GPU 计算完成 → 哈希入 GPU prefix cache(cache_full_blocks)
  → _build_store_jobs 判定可卸载 → prepare_store(FPGA 分配块,块受保护)
  → submit_store → DMA 写 FPGA → complete_store(FPGA 中可 lookup=HIT)
  → 请求结束/逐出 → 新请求前缀命中:
        GPU 有 → 命中 GPU 前缀缓存,不 Load
        GPU 无 + FPGA HIT → prepare_load → submit_load → DMA 读回 → complete_load
```

---

## 6. 错误处理与回退

1. **probe 失败**:按 4.3 回退链处理;全部失败才快速失败(带每个后端的原因)。
2. **传输运行期失败**:`TransferEngine` 捕获异常 → 打 ERROR 日志 → 完成信号照发但标记失败 → 引擎层 `complete_store(success=False)` 交给 manager 处理(现状已有,保留)。
3. **运行期后端失效(可选,Phase 2)**:若连续传输失败达到阈值,冻结该后端,后续 store/load 走回退后端。首版可只记录告警不自动切换,避免复杂度。

---

## 7. 测试策略

### 7.1 无硬件单元测试(本地可跑)

- 用 **mock backend**(继承 `DMABackend`,声明不同能力组合)驱动 `TransferEngine`,验证:
  - `needs_bounce` 真假两条数据流分支;
  - `stream_async` 真假的完成跟踪路径;
  - 上下文保存/恢复逻辑(probe 与注册)。
- `FPGABlockAllocator` / 探测链 / 回退顺序的纯逻辑测试。
- 现有 `tests/v1/kv_offload/fpga/` 保留并扩展。

### 7.2 硬件验证(服务器,需要 FPGA 配置就绪)

- `tmp/verify_bar.c`:验证 GPU↔BAR 数据回环(决定性测试)。
- `tmp/diag_dma_data.py`:Phase A-D 数据正确性。
- `tmp/verify_kv_offload.py`:走 vLLM 完整 store/load 通路,逐字节对比。
- 硬件就绪判据:上述脚本 PASS 后,`probe()` 才会返回 available。

### 7.3 e2e 验证(需要触发真实卸载)

- 短模型 + 长 prompt / 多轮请求,制造 eviction 压力,确认 store/load 实际发生(VLLM_LOGGING_LEVEL=DEBUG 观察 `Store:`/`Load:` 行)。
- 关闭 GPU KV cache 的可用空间(`gpu_memory_utilization` 调小)加速触发。

---

## 8. 配置

`kv_connector_extra_config` 新增/调整:

| 键 | 默认 | 说明 |
|---|---|---|
| `backend` | 无(走探测链) | 显式指定后端名,probe 失败仍回退 |
| `backend_priority` | `["gpu_direct_p2p","intel_fpga","xdma_bounce"]` | 探测优先级,可裁剪 |
| `fpga_bytes_to_use` | 16 GiB | FPGA 容量(保留) |
| `eviction_policy` | `"lru"` | (保留) |
| `xdma_prefix` | `/dev/xdma0` | 仅 XDMA 后端使用(保留) |

环境变量调整:

| 变量 | 状态 | 说明 |
|---|---|---|
| `VLLM_FPGA_BACKEND` | 保留(语义改为"优先") | 显式指定后端 |
| `VLLM_FPGA_DEVICE_ID` | **废弃** | 改由 worker 的 `model_device` 决定,避免 device 不匹配(P3) |
| `VLLM_FPGA_MOCK` | 保留 | 无硬件测试 |

---

## 9. 迁移路径

从现状到新架构,分步演进(每步可独立合并、验证):

1. **`GPUDirectP2PBackend` 重构**:上下文保存/恢复(已做)、device 对齐(已做)、内部去掉 `cuStreamSynchronize` 改事件驱动、BAR 发现从硬编码改为 sysfs 探测。
2. **能力模型落地**:`BackendCapabilities` + 三个后端声明能力;`TransferEngine` 改由能力驱动,删鸭子类型。
3. **探测协议落地**:`probe()` + `ProbeResult`;三后端实现 probe;worker 后端选择改探测链。
4. **`IntelFPGAP2PBackend` 实现**:接入 OPAE/DFL(或 vendor 库)的 `write`/`read` + `probe`。依赖服务器上驱动可用性。
5. **清理**:`gpu_direct_p2p_fixed.py` 合并/删除(两套硬编码 BDF/BAR 应被 sysfs 探测取代);`VLLM_FPGA_DEVICE_ID` 移除。

---

## 10. 风险与开放问题

1. **OPAE/DFL 驱动可用性**:服务器上是否已装 Intel FPGA 驱动、设备节点是否暴露 —— 决定 `intel_fpga` 后端能否真正实现(需服务器确认,`ls /dev/dfl*` / `libopae` 存在性)。
2. **FPGA bitstream 前置**:08-03 已实测 aperture 存活(`diag_dma_data.py` 全 PASS),`gpu_direct_p2p` 通路当前可用。但硬件状态可能再次变化 —— probe 每次启动实时回环验证,状态变了就正确切换可用性,不依赖假设。
3. **IOMEMORY 平台支持面**:`cuMemHostRegister(IOMEMORY)` 在部分 NVIDIA 驱动/GPU 上注册失败;该路径必须始终有回退,不能被当作默认。
4. **带宽目标未量化**:重设计优先保证正确性(回环验证)与多厂商支持;带宽优化(块合并、多流、真异步)作为后续量化迭代。
5. **BAR/BDF 自动发现**:首版可保留配置化 BDF/BAR(probe 时验证),sysfs 自动发现(vendor ID 匹配)作为增强。

---

## 附录 A:与调试证据的对照

| 设计决策 | 调试证据 |
|---|---|
| probe 必须做回环验证 | `verify_kv_offload.py`:store 通路跑通但 FPGA 读回 `0xff`;`diag_dma_data.py` Phase C:纯 host 写读也是 `0xff` |
| 上下文保存/恢复契约 | `cuCtxSetCurrent` 泄漏 → Triton warmup 崩溃(已修) |
| device 与模型一致 | `VLLM_FPGA_DEVICE_ID=1` vs 模型 `cuda:0` → 跨上下文(已修) |
| 按 vendor 探测 + 回退 | 板卡为 Altera/Intel(OPAE/DFL),非 Xilinx XDMA;CvP disabled 无 bitstream |
| 后端能力声明 | `BlockTransferEngine` 用 `hasattr(backend,"bounce_buffer")` 鸭子类型 → 不可扩展 |
