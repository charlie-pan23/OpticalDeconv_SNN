## 总体路线

**目标**：用你训练好的模型，生成 HIPSA 评估所需的全部活动统计量（spike activity、ADC request activity、active SOPs），然后填入 Table 1–5、绘制 Fig.5–7，最终完成 Section 4 的写作。

**依赖关系**：
1. 必须先提取活动迹线 → 才能计算 HIPSA 延迟/能量
2. 必须先测量 CPU/GPU 基线 → 才能画 Fig.5(b)(c)
3. 必须先有活动迹线 → 才能做 ADC pool sweep 和 robustness sweeps

---

## 任务清单（按优先级排序）

### 🔴 任务 1：提取模型活动统计量（最关键，其他任务的基础）

**目标**：获得 CIFAR10-DVS 测试集上每层的：
- 输入 spike activity（每个时间步非零 spike 的比例）
- 每层输出的 spike activity（用于下一层输入）
- 每层 comparator-request activity（即 ADC 请求率，基于膜电位阈值比较）
- 每层的 active SOPs（非零 spike × 权重非零的乘积数）

**方法**：
- 在你的训练脚本基础上，写一个 **inference 模式**，遍历测试集（batch size=1），在模型的每个 LIF 层后记录：
  - 输入 spike 张量的非零比例（`(x>0).float().mean().item()`）
  - 输出 spike 张量的非零比例
  - 对于 ADC request activity：可以近似为**输出膜电位中绝对值大于某阈值的比例**（阈值可取 0.5 或根据你的模型实际阈值设定）。如果模型输出层用了 `reset_mechanism="none"`，ADC request 可以理解为需要转换的 partial sum 数量，即 HAPR 后需要 ADC 的通道数。
- 统计所有样本的平均值，得到全局的 `input_spike_activity`、`output_spike_activity`、`ADC_request_activity`、`active_SOPs_per_image`。

**输出**：
- 每个类别的平均统计量（至少给出 CIFAR10-DVS 整体的值）
- 用于填充 Table 1 中的 “Main input spike activity”、“Main ADC request activity”、“Main cycles per image”、“Main ADC requests per image”。

**提示**：
- 你当前的模型有 8 个 LIF 层（lif1~lif8），建议逐层记录。
- 计算 active SOPs：对于卷积层，active SOPs = 输入非零 spike 数 × 输出通道数 × 卷积核尺寸（如果权重已量化，还需乘以非零权重比例）。你的模型权重是浮点，可以认为所有权重非零，所以 active SOPs ≈ 输入非零 spike 数 × 输出通道数 × 9（3×3 conv）或 × 1（FC）。
- 更精确的做法：在 forward 中插入 hook，统计每层实际发生的乘法次数（`torch.count_nonzero(x) * kernel_elements`）。

**预估工作量**：半天到一天。

---

### 🟠 任务 2：测量 CPU/GPU 基线性能

**目标**：获得 CPU 和 GPU 上运行同一模型（batch size=1, T=10）的：
- 端到端延迟（ms）
- 活跃功耗（W）和空闲功耗（W）
- 能量/图像（mJ）

**方法**：
- CPU：使用 PyTorch CPU 后端，在 Intel i5/i7 或类似平台上运行测试集的一个子集（例如 100 张），测量总时间，减去空闲功耗。
- GPU：使用 PyTorch CUDA 后端，在 RTX 4050/4070 上运行，使用 `torch.cuda.Event` 同步计时，并用 `nvidia-smi` 或 `pynvml` 读取功耗。
- 注意：预热后测量，排除模型加载和磁盘 I/O。

**输出**：
- 填入 Fig.5(b)(c) 的 CPU/GPU 柱状图。

**提示**：
- 如果你没有功耗测量工具，可以先用典型值估算（CPU 50W，GPU 80W），但论文中最好注明是估算值。
- 延迟可以直接测量，能量 = 延迟 × (活跃功耗 - 空闲功耗)。

**预估工作量**：半天。

---

### 🟡 任务 3：计算 HIPSA 延迟和能量

**目标**：基于任务 1 得到的 active SOPs 和 HIPSA 参数，计算：
- 延迟：`latency = active_SOPs_per_image / 6.55 TSOP/s`
- 能量：`energy = latency × P_total`，其中 `P_total` 来自 Table 4（2.424 W）
- 效率：`efficiency = 6550 / P_total`（GOPS/W）

**方法**：
- 直接代入公式。注意 active SOPs 需要换算成 **active SOPs per image**（即所有时间步的总和）。
- 你模型的总 active SOPs 大约在 4.7×10^8 量级（与文档一致），用 6.55 TSOP/s 可得延迟约 0.072 ms，吞吐约 13950 image/s。

**输出**：
- 填入 Fig.5(b)(c) 的 HIPSA 柱状图，以及 Table 3 的 HIPSA 行。

**预估工作量**：半小时（纯计算）。

---

### 🟢 任务 4：填写 Table 1–5 和绘制 Fig.6

**目标**：
- **Table 1**：填入你实际的工作负载参数（CIFAR10-DVS, T=10, batch=1, 精度等）以及任务 1 得到的 activity 值。
- **Table 2**：设备参数已由文档给出，你只需确认是否与你的设计一致（应一致）。
- **Table 3**：需要查找并填入其他加速器的数据（FPGA/ASIC/Photonic）。文档中有部分占位符，你需要从参考文献中搜集真实数据。注意 metric definition 列要区分 active SOPs 还是 dense MACs。
- **Table 4**：功率和面积分解。文档已给出详细计算，你可以直接采用，但需确认你的 HIPSA 配置是否完全匹配（4 tiles, 64×64, 16 ADCs, 32 HAPR lanes 等）。如果不匹配，需调整宏计数。
- **Table 5**：MRR 锁定敏感性。文档给出了两种情形，你可以直接引用。
- **Fig.6**：
  - (a) 功率分解饼图：用 Table 4 的数据画。
  - (b) ADC pool-size sweep：需要模拟不同 ADC 数量（8/16/32/64）下的能量、利用率和仲裁停顿率。这需要你建立一个简单的队列模型或使用 HIPSA 的 cycle-accurate 模拟器。如果时间不够，可以先做一个理论估算（假设线性缩放）。
  - (c) MRR 稳定性敏感性：直接用 Table 5 的数据画柱状图。

**提示**：
- Table 3 的对比数据可以从 PICoSNN 论文（Optical SNN.pdf）的 Table III 中摘取部分，但注意其工作负载和精度可能与你的不同，需要注明。
- 如果无法获得其他加速器的确切数据，可以标注“under review”或“estimated from reported specs”。

**预估工作量**：1–2 天（主要是搜集数据和画图）。

---

### 🔵 任务 5：绘制 Fig.7 鲁棒性分析

**目标**：展示在不同光电器件扰动下的精度和活动变化。

**方法**：
- 你需要一个 **硬件感知推理模拟器**，能够在模型推理过程中注入扰动：
  - MRR 传输扰动（0%, 1%, 2%, 3%, 5%）
  - WDM 串扰（-30, -25, -20, -15 dB）
  - 激光强度波动（0%, 1%, 2%, 3%）
  - ADC 精度（4, 5, 6, 8 bit）
  - 比较器阈值（0.01, 0.02, 0.05, 0.10 full scale）
  - HAPR group size（G=4, 8, 16）
- 最简单的方法：在模型的权重或激活上添加高斯噪声来近似扰动（如文档中所述）。但文档要求“device-specific perturbation sweeps”，所以最好能模拟物理效应，例如对 MRR 权重乘上一个传输因子，对 ADC 输出进行量化等。
- 你可以先实现一个简化的版本：对权重加乘性噪声（模拟 MRR 误差），对激活加加性噪声（模拟串扰），然后观察精度变化。

**输出**：
- Fig.7(a)：MRR 扰动、WDM 串扰、激光波动下的精度曲线。
- Fig.7(b)：ADC 精度、比较器阈值、HAPR group size 下的精度、ADC 活动、能量。

**提示**：
- 如果时间紧张，可以先只做 CIFAR10-DVS 的 MRR 扰动和 ADC 精度扫描，作为主要结果。
- 可以使用你现有的 inference 脚本，在 forward 中插入扰动操作。

**预估工作量**：2–3 天（包括编写模拟器和运行实验）。

---

## 时间规划建议

| 周次 | 任务 | 产出 |
|------|------|------|
| 第1周 | 任务1（提取活动统计）+ 任务2（CPU/GPU基线） | 得到所有活动值和基线数据 |
| 第2周 | 任务3（HIPSA延迟/能量计算）+ 任务4（填写表格和画图） | Table 1–5, Fig.5, Fig.6 初版 |
| 第3周 | 任务5（鲁棒性模拟） | Fig.7 初版 |
| 第4周 | 整合、润色、补充对比数据、撰写 Section 4 文本 | 完整的 Section 4 草稿 |

---

## 需要你编写的代码模块

1. **活动统计提取脚本**：基于 `train_cifar10dvs.py` 或 `snn_vgg.py`，添加 hook 或直接在 forward 中记录每层 spike 和 membrane 统计量。
2. **CPU/GPU 基准测试脚本**：简单推理循环 + 计时 + 功耗读取（可选）。
3. **硬件感知模拟器**：在推理过程中注入扰动，并记录精度。可以复用你的 `CIFAR10DVSDataset` 和 `SpikingVGG` 模型，在 forward 中修改权重或激活。

---

## 需要你手动搜集的资料

- 其他加速器的性能数据（Table 3）：从 PICoSNN 论文、Lightening-Transformer、SpikeX 等文献中摘取。
- 设备参数（Table 2）：已由 HIPSA 文档给出，但你需要确认你的设计是否使用相同的器件（如 MZM vs EAM，ADC 型号等）。

---

## 最后提醒

- **文档1中的占位符**（如 `[Reported]`）都需要你用真实数据替换。
- **不要直接复制 PICoSNN 的参数**，除非你的 HIPSA 设计与其完全相同（显然不同）。HIPSA 是你们组的架构，有自己的参数。
- **所有仿真依赖的值**（如 ADC pool sweep 中的 stall rate）如果无法精确模拟，可以在论文中说明是 analytical estimate 并注明假设。

如果你需要我帮你编写上述任何一个模块的代码（例如活动统计提取脚本、硬件感知模拟器），随时告诉我，我可以直接给你可运行的 Python 代码。