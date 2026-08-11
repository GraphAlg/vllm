# FPGA KV Cache Offload — 学生任务对齐手册

> 面向接手该项目的学生。目标:你在 **20 分钟**读完第一部分就能知道系统长什么样,**按第二部分的任务清单**就能开始干活,**用第三部分的脚本**自测。
> 配套设计文档:[2026-08-01-fpga-kv-offload-transport-design.md](2026-08-01-fpga-kv-offload-transport-design.md)

---

## Part 1 · 架构速览(20 分钟)

### 1.1 这个项目在做什么

在 vLLM 推理时,把 KV cache 从 GPU 显存**卸载到 FPGA 扩展内存**,从而扩大可用 KV 容量(支持更长上下文 / 更多并发)。卸载不是"挪走",而是**在 FPGA 里多存一份副本**,GPU 显存不够时才读回。

### 1.2 分层架构(记住这张图)

```
OffloadingSpec (FPGAOffloadingSpec)     容量/块大小计算
  └─ OffloadingManager (FPGAOffloadingManager)   块跟踪 + LRU/ARC 逐出
      └─ OffloadingWorker (FPGAOffloadingWorker)  异步传输调度
          └─ TransferEngine                       后台线程串行化传输
              └─ DMABackend (抽象)                vendor 无关传输接口
                  ├─ GPUDirectP2PBackend           GPU DMA 直写 FPGA BAR
                  ├─ XDMABounceBufferBackend       XDMA + host 反弹缓冲(回退)
                  └─ IntelFPGAP2PBackend           Intel 驱动(骨架)
```

**两类角色**:调度侧(engine core,决定"何时卸载")和 worker 侧(model runner,执行"怎么卸载"),中间用 `OffloadingConnectorMetadata` 传 job。

### 1.3 一次完整的卸载(store)流程

```
1. 请求 prefill 算 token → 块写满 → GPU 前缀缓存打哈希(链式哈希:块N哈希=hash(块N-1哈希+token))
2. 调度侧每步:_build_store_jobs 判定哪些块可卸载
   触发条件: 已计算token数 // 块大小 > 已卸载块数
3. manager.prepare_store(keys) → 分配 FPGA 块(满则 LRU/ARC 逐出)
4. worker 侧:submit_store → TransferEngine 后台线程 → backend.write → FPGA
5. complete_store → 该块在 FPGA 可被 lookup 到
```

### 1.4 关键文件地图

| 文件 | 职责 | 学生重点看 |
|---|---|---|
| `vllm/v1/kv_offload/fpga/spec.py` | 容量/块数计算 | 块大小怎么算 |
| `vllm/v1/kv_offload/fpga/manager.py` | 块跟踪,委托 CPU 逻辑 | prepare_store/load 语义 |
| `vllm/v1/kv_offload/fpga/worker.py` | 传输调度 + 后端选择 | **后端探测链(任务4要改)** |
| `vllm/v1/kv_offload/fpga/copy_backend.py` | 传输引擎,后台线程 | **能力驱动分支(任务3要改)** |
| `vllm/v1/kv_offload/fpga/backend_base.py` | DMABackend 抽象 | **能力声明 + probe(任务2/4)** |
| `vllm/v1/kv_offload/fpga/backends/*.py` | 三种物理通路 | gpu_direct_p2p 是当前主力 |
| `vllm/v1/kv_offload/base.py` | offload key 定义 | block_hash + group_idx |
| `.../offloading/scheduler.py` | 调度侧何时卸载 | `_build_store_jobs`/`_build_load_jobs` |
| `.../offloading/worker.py` | job → 实际传输 | store 推迟机制 |

### 1.5 术语表

| 术语 | 含义 |
|---|---|
| store / load | GPU→FPGA / FPGA→GPU |
| offload key | `block_hash + group_idx`,FPGA 里块的身份 |
| offloaded_block_size | 一个卸载块含多少 token(≥ GPU 块) |
| offload_prompt_only | 只卸载 prompt 块(默认 True) |
| bounce buffer | host 上的一块钉住内存,两跳路径的中转站 |
| DMABackend | 一种物理传输方式(XDMA/P2P/Intel) |

---

## Part 2 · 任务清单(可分配)

> 每个任务:目标 / 涉及文件 / 验收标准 / 依赖。**验收标准就是老师检查你工作的标准** —— 你按它自测,能过就是完成了。
> 涉及文件给出的是"入口",具体读哪些从入口往调用链展开。

### 任务 0:环境与基线(0.5 天)—— 所有人先做

- **目标**:在服务器上跑通现有验证脚本,建立"现在系统是好的"基线
- **做**:
  1. 处理 FlashInfer JIT 报错(两种方式之一:服务器装 CUDA12 工具链;或跑时加 `VLLM_USE_FLASHINFER_SAMPLER=0`)
  2. 跑 `tmp/verify_kv_offload.py`(验证 store/load 数据回环)
  3. 跑 `tmp/diag_dma_data.py`(数据正确性 Phase A-D)
- **验收**:`verify_kv_offload.py` 输出 `STORE VERIFY: PASS` 和 `LOAD VERIFY: PASS`
- **依赖**:无

### 任务 1:现有传输层代码走读(1 天)—— 所有人先做

- **目标**:能不看代码给同学讲清 store/load 全链路
- **做**:读 Part 1 文件地图里的所有文件,画数据流图,写一份理解笔记
- **验收**:能回答"一次 store 从调度决策到 FPGA 物理写入经过了哪几层、各层做什么"
- **依赖**:任务 0

### 任务 2:能力模型 BackendCapabilities(0.5 天)

- **目标**:定义后端能力数据结构,三个后端各自声明能力
- **做**:在 `backend_base.py` 加 `BackendCapabilities`(字段见设计文档 4.1:`stream_async`/`needs_bounce`/`max_transfer_bytes`/`alignment`);三个 backend 各声明一份;写单元测试
- **验收**:每个后端有明确的 capabilities;测试覆盖字段;设计文档 4.1 的对照表和你实现一致
- **依赖**:任务 1

### 任务 3:TransferEngine 能力驱动重构(1-2 天)

- **目标**:删掉 `hasattr(backend, "bounce_buffer")` 鸭子类型,改用 `capabilities.needs_bounce`
- **做**:改 `copy_backend.py`,store/load 分支由能力驱动;用 mock backend(声明不同能力)验证两条分支
- **验收**:mock 下 `needs_bounce=True/False` 两条路径都走通;`verify_kv_offload.py` 不回归
- **依赖**:任务 2

### 任务 4:探测协议 probe() + 回退链(1-2 天)

- **目标**:每个后端能自检"我是否可用",worker 按优先级探测选择
- **做**:在 `backend_base.py` 加 `ProbeResult` + 类方法 `probe()`;三后端实现(**probe 必须做真实数据回环验证**,不能只看文件存在);改 `worker.py` 后端选择为探测链
- **验收**:探测失败正确回退到下一后端;日志清晰;没有 FPGA/驱动时 probe 正确返回 unavailable(而不是崩)
- **依赖**:任务 2
- **关键**:这是整个设计最重要的部分,做之前重读设计文档 4.2 / 4.3 和附录 A

### 任务 5:异步化(1 天)

- **目标**:去掉后端内部 `cuStreamSynchronize`,传输不再阻塞等待
- **做**:改 `gpu_direct_p2p.py` 的 `_copy_and_wait`,改为只下发到 stream + 引擎用 CUDA event 跟踪
- **验收**:`verify_kv_offload.py` 仍 PASS;观察日志确认传输与计算可并行
- **依赖**:任务 1(需要理解现有 event 机制)

### 任务 6:gpu_direct_p2p 重构 + BAR 探测(1-2 天)

- **目标**:上下文保存/恢复契约(已实现,保留)、device 对齐(已实现,保留)、BAR 从硬编码改为 sysfs 探测
- **做**:改 `gpu_direct_p2p.py` 的 BAR 发现(读 `/sys/bus/pci/devices/<bdf>/resource*`,验证 aperture 不是 0xff);配合任务 4 的 probe
- **验收**:probe 能区分"aperture 活"与"aperture 死(读回 0xff)";不同 BDF/BAR 都能正确发现
- **依赖**:任务 4
- **背景**:硬编码 `FPGA_PCI_BDF = "0000:88:00.0"` 和 `FPGA_BAR_INDEX = 2` 只在当前板卡有效

### 任务 7:Intel OPAE 后端(2-3 天,依赖服务器驱动)

- **目标**:把 `intel_fpga.py` 从骨架实现为可用后端(OPAE/DFL 驱动)
- **做**:实现 `write`/`read` + `probe`;接入 Intel 驱动用户态库
- **验收**:驱动可用时数据回环 PASS(参考 `verify_kv_offload.py` 的思路);驱动不可用时 probe 正确返回 unavailable
- **依赖**:任务 4;服务器上有 Intel 驱动(`ls /dev/dfl*` / libopae 存在)

### 任务 8:清理与文档(0.5 天)

- **目标**:删残留,收尾
- **做**:合并/删除 `gpu_direct_p2p_fixed.py`(两套硬编码 BDF/BAR 应被任务 6 的探测取代);移除 `VLLM_FPGA_DEVICE_ID`;更新设计文档/注释
- **验收**:无遗留硬编码 BDF/BAR;env 清理干净;文档与代码一致
- **依赖**:任务 6

### 可选任务(有余力再做)

| 任务 | 说明 | 依赖 |
|---|---|---|
| 9: huge page bounce | bounce buffer 用 2MB 页分配,提升两跳路径带宽(机制见 Part 4) | 任务 3 |
| 10: GDR 后端 | GDRCopy 让 FPGA 直读 GPU 显存(单跳替代 IOMEMORY) | 任务 4 + 服务器 GDRCopy |

---

## Part 3 · 验证方法(自测手段)

> 每次改动后,按此顺序自测。**先小后大**:先回环脚本,再 e2e。

### 3.1 数据回环(改传输层必跑)

```bash
# 1) 裸 BAR 数据正确性(C 级,不依赖 vLLM)
sudo fpga-env/bin/python tmp/diag_dma_data.py
# 期望:Phase A-D 全部 True

# 2) vLLM 完整 store/load 通路
sudo fpga-env/bin/python tmp/verify_kv_offload.py
# 期望:STORE VERIFY: PASS + LOAD VERIFY: PASS
```

### 3.2 端到端推理(验证调度侧真的卸载)

```bash
sudo -E env VLLM_LOGGING_LEVEL=DEBUG VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  fpga-env/bin/python test_fpga_infer.py 2>&1 | grep -E "Store:|Load:|Error|Traceback"
```

- 期望:`Store: N GPU blocks → FPGA (backend=GPU_DIRECT_P2P)`
- 短 prompt 可能不触发(不足一块);用长 prompt(几百 token)或设 `offload_prompt_only: false` 强制触发
- 出现 `Load:` 需要 GPU 逐出 + 复用前缀(参考设计文档 5.1)

### 3.3 无硬件环境(本地 mock)

- `VLLM_FPGA_MOCK=1` + `xdma_bounce` 后端,可在没有 FPGA 的机器上跑通传输逻辑
- 用于开发期快速迭代,硬件留给验收

---

## Part 4 · 关键概念补充(老师讲过就跳过)

### 4.1 为什么 offload 能扩容却不拖慢推理

- 卸载是**异步后台**的:store 发出去后 GPU 继续算,不阻塞 token 生成
- store 故意**推迟到下一步开头**提交,避开当步的采样
- 但要注意:传输带宽是瓶颈 —— 需要 **大块合并 + 真异步**,这正是任务 3/5 要做的

### 4.2 链式哈希与前缀复用

块 N 的哈希 = `hash(块N-1哈希 + 块N的token)`。两个请求共享前 K 个块 ⇔ 前 K 个哈希相同。offload key 就是"哈希 + 组号"。**前缀复用在 FPGA 里能否成立,取决于这个哈希设计**。

### 4.3 两跳路径为什么慢 + huge page 怎么帮

- XDMA/OPAE 是 scatter-gather DMA,4KB 页时 SG 表项巨大(1GB ≈ 26 万项)
- 2MB huge page 把项数降到 1/512 → 带宽提升
- 这是任务 9 的动机;只影响 bounce 路径,GPU 直连不受影响

### 4.4 为什么必须"回环验证"而非"注册成功"

实测:即使 `cuMemHostRegister` 注册成功,aperture 也可能读回全 `0xff`(FPGA 未配置/未解码地址)。所以后端可用性必须靠**写已知数据 → 读回 → 对比**判定,不能只看注册/文件存在。这是任务 4 的核心要求。

---

## Part 5 · 工作方式建议(给学生的)

1. **改前先跑基线**:任务 0 的脚本跑一遍,确认"现在是好的"
2. **一次只改一个任务**:每个任务有独立验收,过了再动下一个
3. **小改动先 mock 再真机**:开发期用 mock,验收再上真机
4. **读代码从入口跟调用链**:不要从头读到尾,从 `submit_store` 往下跟一次完整调用
5. **卡住了先看设计文档**:架构决策、契约、验收标准都在里面,比代码注释全

---

## 附录:任务依赖图

```
任务0(环境基线)
  └─ 任务1(代码走读)
       ├─ 任务2(能力模型)
       │    ├─ 任务3(引擎能力驱动)
       │    │    └─ 任务5(异步化)
       │    └─ 任务4(probe+回退链)
       │         ├─ 任务6(BAR 探测)
       │         │    └─ 任务8(清理)
       │         └─ 任务7(Intel OPAE)
       └─ (可选)任务9 huge page / 任务10 GDR
```
