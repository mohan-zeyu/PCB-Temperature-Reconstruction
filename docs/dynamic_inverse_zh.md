# 动态热反演模块（dynamic_inverse）

本文档说明 `tdp/dynamic_inverse/` 的物理模型、可辨识边界、数据流程与使用方法。
阅读顺序建议：第 1 节（能报什么、不能报什么）→ 第 3 节（运行方法）→ 其余按需。

---

## 1. 物理模型与可辨识边界

### 1.1 从有量纲方程到可辨识形式

薄板动态热传导：

```text
theta(x,y,t) = T(x,y,t) - T_ambient(t)      过余温度            [K]
C(x,y)       = rho * cp * d                  面热容              [J/(m^2*K)]
G(x,y)       = k_eff * d                     面热导              [W/K]
q(x,y,t)                                     面功率密度          [W/m^2]

C * d(theta)/dt = div(G * grad(theta)) - h * theta + q
```

两边同除 `C`：

```text
d(theta)/dt = div(alpha_eff * grad(theta)) - beta * theta + s

alpha_eff = G / C     有效热扩散率
beta      = h / C     冷却速率系数
s         = q / C     归一化热源速率
L         = sqrt(alpha_eff / beta)   热扩散特征长度
```

### 1.2 全局尺度不可辨识性（本项目的核心限制）

把 `C, G, h, q` 同时乘以任意常数 `lambda`，`alpha_eff`、`beta`、`s` 都不变，
因此**任何温度轨迹都完全相同**。只有温度序列的实验无法区分这些板子。

结论：**不得输出、不得声称**以下量：

| 量 | 单位 | 为何不可辨识 |
|---|---|---|
| 像素级材料热导率 `k(x,y)` | W/(m·K) | 需要 `rho*cp*d`，本实验未测 |
| 绝对面热导 `G(x,y)` | W/K | 同上 |
| 绝对功率密度 `q(x,y,t)` | W/m² | 同上 |
| 元器件绝对功率 | W | 同上 |
| 隐藏铜层真实几何 | — | 表面红外无法揭示内层结构 |

**可以**输出的量（本模块实际输出）：

| 量 | 符号 | 单位（有时间戳 / 无时间戳） |
|---|---|---|
| 有效热耦合速率 | `a_ij` | `1/s` / `1/frame` |
| 有效冷却速率 | `beta_i` | `1/s` / `1/frame` |
| 有效时间常数 | `tau_i = 1/beta_i` | `s` / `frame` |
| 有效热扩散率 **proxy** | `alpha_proxy_i` | `px^2/s` / `px^2/frame`（有像素尺寸时附加 `mm^2/s`） |
| 热扩散特征长度 | `L_i` | `px`（附加 `mm`） |
| 归一化热源速率 | `s_i(t)` | `K/s` / `K/frame` |
| 相对热源份额 | `relative_source_i(t)` | 无量纲 |

`alpha_proxy` 由图耦合速率与区域中心距构造：`alpha_proxy_i = <a_ij * d_ij^2>`，
在均匀网格极限下退化为连续场的 `alpha`。**它是 proxy，不是材料热导率。**

### 1.3 AI 不能创造实验中不存在的信息

神经网络只用来预测**物理 ODE 的系数**（边权、冷却率、源），
轨迹始终由积分该 ODE 得到。增加网络容量不会让绝对功率变得可辨识；
纯黑箱下一帧预测只作为对照基线，不作为主结果。

---

## 2. 本次实验的具体限制

### 2.1 相机发射率设置未记录

2026-07-26/27 两次采集导出的中文格式 CSV **不含参数块**，
因此相机当时设定的发射率 `eps_set` 与反射温度**没有被记录**。
2026-07-09 的英文格式 CSV（case06/07）记录了 `Emissivity: 0.93, Reflected Temp.: 25.0`。

处理方式：在 `configs/measurements_v3.yaml` 中把 `assumed_emissivity: 0.95`
显式标记为 `recorded: false`，并提供 `sensitivity_sweep`。
`run_metadata.json` 中如实写明这是假设而非测量。

### 2.2 没有真正的断电冷却区间

`记录csv文件/cpu_load_100.csv` 显示 case10 的 100% 负载在 16:52:11 开始、
约 17:17 结束；`soc_log_100.csv` 显示此后 SoC 温度稳定在 46–48 °C，
**远高于环境温度**——负载停止了，板子并没有断电。

因此：

* 该区间在所有输出中标记为 `free_cooling_candidate`，**从不写作 free cooling**；
* 拟合热算子时**不假设 `q=0`**，而是把残余空载源当作一个**常数、非负**的
  待拟合量 `s_idle`（初值为 0），与 `a_ij`、`beta_i` 联合辨识；
* 由 `0 = A * theta_idle + s_idle` 反解得到的 `theta_idle` 作为诊断输出。

> 关于实现的一个重要教训：另一种等价写法是先估计 `theta_idle`、再拟合
> `psi = theta - theta_idle`。**这在数值上是错的**，除非 `theta_idle` 恰好是
> 真正的稳态：否则 `d(psi)/dt = A*psi + A*theta_idle`，多出来的常数项会污染
> 所有速率。在合成数据上这一处错误会把边权相关系数从 0.89 打到负值。
> 见 `tdp/dynamic_inverse/train_thermal.py` 中的注释与
> `tests/test_synthetic_recovery.py::test_residual_idle_source_is_near_zero_when_the_system_truly_free_cools`。

### 2.3 没有可见光 RGB 图

`case*/photo/*.jpeg` 是相机对**同一帧热数据**的伪彩渲染（带屏显刻度与帧计数），
不是可见光照片。因此多模态分割只能使用
「温度动态特征 + 反演发射率」，不能使用 RGB。
`RegionPartition.report["rgb_available"]` 恒为 `false` 并附说明。

### 2.4 像素物理尺寸是几何估计

相机未记录焦距/视场。像素尺寸由**树莓派 4B PCB 外形 85 × 56 mm**
重采样到规范板坐标网格 `157 × 103` 得到：

```text
0.5414 mm/px (行)   0.5437 mm/px (列)   取各向同性 0.5425 mm/px
```

`run_metadata.json` 中 `pixel_size_calibrated: false`。
`mm` 单位的结果始终与 `px` 单位并列给出。

### 2.5 传导与对流的可分性（本次最重要的方法学发现）

`a_ij` 与 `beta_i` 只通过每个区域的**总弛豫率**进入动力学：

```text
d(psi_i)/dt = -(beta_i + sum_j a_ij) * psi_i + sum_j a_ij * psi_j
```

若整板**自相似降温**（各区域过余温度按同一比例下降），第二项恒为第一项的固定倍数，
传导就不留下任何可观测痕迹，拟合可以令 `a=0` 并把一切吸收进非均匀的 `beta`。

`tdp/dynamic_inverse/identifiability.py` 用**完全无模型**的方法测量这一点：
归一化剖面 `psi_i(t)/mean(psi(t))` 的空间标准差随时间的变化。
case10 的实测结果：

```text
初始剖面离散度 0.271  ->  95 s 后settle到 0.108   （降低 2.5 倍）
携带传导信息的帧数：171 帧中约 15 帧（8.8%）
直接拟合的区域总弛豫率：中位 5.79e-3 1/s（tau = 173 s），p90/p10 = 1.73
仅靠 beta 非均匀性复现该剖面塌缩所需的速率离散度：9.7e-3 1/s
实测速率离散度（p10-p90）：3.0e-3 1/s  ->  不足以解释
```

**结论：该降温区间确实携带传导信息，但集中在卸载后最初约 95 秒。**

这一诊断同时解释了三次失败的拟合尝试（均已修正，见代码注释）：

1. 边权头与冷却头共用一个初始尺度时，边权需要跨越约一个数量级才能到位，
   Adam 在早停前走不到；
2. 改用 `theta/delta_theta` 推出的耦合尺度则**高估约一个数量级**——
   图拉普拉斯对邻域差分求和时正负抵消，净值远小于平均绝对差——
   结果边权被梯度推到 softplus 的饱和负尾，拟合出 `a = 1e-10`；
3. 改用 `(1/t_settle - beta) / (2 * mean_degree)` 又**低估约 30 倍**：
   横跨整板的形变模态远慢于单条边的交换速率，除以局部配位数是错的。

**正确的换算除以图拉普拉斯的 Fiedler 值**（最小非零特征值），
即该形变模态本身的特征值。但这一反演是**上界**：实测的剖面塌缩里还混有
传感器噪声与热源关断的贡献，把它全部记到传导头上会高估。
而"耦合与冷却同量级"是**下界**。最终取二者的几何平均：

```text
a_upper ~ (lambda_shape - beta) / lambda_2(L_graph)
a       ~ sqrt(a_upper * beta)
```

见 `train_thermal.coupling_rate_scale` 与 `graph_fiedler_value`。
case10 上 `mean_degree/lambda_2 = 30.5`，正是"除以配位数"与
"除以 Fiedler 值"两种取法相差的倍数。

边权头**从该尺度的 0.3 倍起步**（`model.ThermalOperator.edge_init_fraction`）：
softplus 两个方向的失效并不对称——起点偏低还能往上爬，
起点偏高会被梯度推进负饱和尾而回不来。宁可从下方逼近。

该配置在 6 组（3 个合成数据种子 × 2 个拟合种子）上的恢复表现：
`r_edge = 0.69–0.84`、强耦合边召回 `0.71–0.86`、`a` 对比度 `3.4–4.4`（真值 5）、
`beta` 周边/内部比 `1.34–1.60`（真值 3.6）。
即**方向可靠、对比度被压缩**——后者作为限制写入报告，而不是把测试改松。

**验证方式是完全无模型的径向剖面**：稳态下以最热点为中心测量
`theta(r)`，实测 0–42 mm 范围内 `theta` 在 29–36 K 之间（起伏 <±10%），
即**整板接近等温**，因此真实扩散长度必须远大于板尺寸，
而不是修正前拟合出的 3 mm。修正后 `L ~ 23 mm`，与观测自洽。

修正后结果对 steady 先验权重（0 / 0.3 / 1 / 3）**几乎完全不敏感**
（`alpha` 变化 < 1%），说明该值由数据而非先验决定。

### 2.6 近等温板：耦合的空间对比度不可恢复

传导只在归一化剖面仍在变化时可见；剖面稳定后整板进入基模，
无论底下的耦合如何，每个区域都报告同一个速率。
因此关键量不是记录总长度，而是**分辨该形变模态的帧数**。

用合成数据（真值已知）标定该阈值：

| 情形 | 分辨形变模态的帧数 | 强耦合边召回 | `a` 量级 |
|---|---|---|---|
| `a/beta ~ 2.4` | 14 | 0.71–0.86 | 基本正确 |
| `a/beta ~ 15`（近等温） | 7 | **0.29（≈随机）** | **低 5–15 倍** |

`MIN_INFORMATIVE_FRAMES = 12` 即由此标定
（`tdp/dynamic_inverse/identifiability.py`）。
**case10 实测为 15 帧**，落在可恢复一侧——因此 `a_ij` 的*排序*有意义，
但其*绝对量级*仍是本项目中最弱的量。
`tests/test_synthetic_recovery.py::test_audit_flags_the_regime_where_coupling_contrast_is_unrecoverable`
同时断言"审计给出警告"与"拟合确实失败"，保证该警告是标定过的而非装饰。

### 2.7 alpha_proxy 是 proxy，且依赖分割尺度

`alpha_proxy_i = <a_ij * d_ij^2>` 建立在**不规则 SLIC 镶嵌**上的图拉普拉斯之上，
不像 `beta`（区域自身属性）那样天然与分割尺度无关。
`uncertainty.partition_ensemble` 在 48/64/96 三个分辨率上重拟合并报告
`alpha_resolution_spread`，以区分"板子的性质"与"网格的性质"。
不要把 `alpha_proxy` 直接与材料热扩散率比较。

### 2.8 低发射率修复的固有上界

`tdp/data/emissivity.py` 的传导填补由**footprint 周围可信像素**驱动。
齐次方程的解不可能在内部产生极大值，因此**有源封装（SoC）的填补是下界**。
本实验中 case10（裸板）SoC 修复到约 64 °C，而同板同负载的 case17（哑光黑漆）
实测约 78 °C——差约 14 K。该区域的 `beta` 与 `a_ij` 因此带有系统偏差，
在报告与不确定性图中标注为低可信。
（`repair_frame` 支持 `anchors` 传入片上温度，默认关闭。）

**这一上界会传播到热源反演。** case10（裸板）反演出的最强热源区域位于
板左上角实测最热点附近（距离约 3 mm），而**不是** SoC——因为 SoC 的传导填补
偏低，而左上角的小元件是高发射率区域、读数本身可信。
换言之：热源图忠实反映了**它拿到的温度场**，而该温度场在 SoC 处系统性偏低。
case17（哑光黑漆）正是为此设置的对照：那里 SoC 读数无需修复，
两者的热源排序差异直接量化了这一偏差。报告中必须同时给出两条序列。

**另一个必须一并说明的事实：热源的集中程度取决于耦合量级，
而耦合量级正是本项目中最弱的可辨识量（见 2.6）。**
在耦合取较强的一档时，case17 的热源前三名区域距最热点 3.6 / 5.9 / 10.0 mm，
合计占 23.6%（均匀值 4.7%），即准确落在 SoC 上；
在耦合取较弱的一档时，热源图退化为温度图的缩放版
（`effective_regions` 从 33.5/64 升到 59/64）。
因此"热源定位"这一结论必须与所用的耦合量级一同引用，
不能脱离 `identifiability_audit` 单独陈述。

---

## 3. 运行方法

```bash
# 单帧数据质量 + 阶段检测 + 区域分割
python -m tdp.dynamic_inverse prepare      --config configs/dynamic_inverse.yaml \
                                           --sequence case10_rpi4b_bare_full

# 断电候选区间上辨识热算子（a_ij, beta_i, 残余空载源）
python -m tdp.dynamic_inverse fit-thermal  --config configs/dynamic_inverse.yaml \
                                           --sequence case10_rpi4b_bare_full

# 冻结热算子，反演通电区间的归一化热源
python -m tdp.dynamic_inverse fit-source   --config configs/dynamic_inverse.yaml \
                                           --sequence case10_rpi4b_bare_full \
    --checkpoint outputs/dynamic_inverse/case10_rpi4b_bare_full/latest/thermal_operator.pt

# 基线阶梯、消融与不确定性
python -m tdp.dynamic_inverse evaluate     --config configs/dynamic_inverse.yaml \
                                           --sequence case10_rpi4b_bare_full

# 汇总 report.md / metrics.json / run_metadata.json
python -m tdp.dynamic_inverse report \
    --run-dir outputs/dynamic_inverse/case10_rpi4b_bare_full/latest

# 一条龙
python -m tdp.dynamic_inverse run --config configs/dynamic_inverse.yaml \
                                  --sequence case10_rpi4b_bare_full
```

所有子命令支持 `--device cpu|cuda|mps`、`--seed`、`--output-dir`、`--dry-run`。
`--smoke` 用极小配置（少帧、少区域、少 epoch）在 CPU 上跑通全链路，约 15 秒。

---

## 4. 数据流程

```text
tdp.io.hm_csv.read_hm_csv           解析 HM*.csv（自带统计量自检）
        |
tdp.data.crop                        rotated_rect_corners + rectify_quad
        |                            （矫正后半程"拍歪了"的板子）
        v
tdp.dynamic_inverse.data.load_sequence
        |  - 板窗口：角度由投影锐度自动估计；PCB 四边由配置给出（去旋转坐标系）
        |  - 漂移配准：梯度 ZNCC，仅当位移 > 1.5 px 才校正（case17 有真实 3 px 漂移）
        |  - 发射率修复：mask 在参考帧上检测一次，全序列复用
        |  - trust_weight = exp(-dist_to_trusted / 6px)
        v
ThermalSequence(temperature_c[T,H,W], valid_mask, trust_weight, emissivity,
                timestamps_s, ambient_c, pixel_size_mm, metadata)
        |
tdp.dynamic_inverse.phases.detect_phases    heating / near_steady /
        |                                    free_cooling_candidate / invalid_transition
        v
tdp.dynamic_inverse.regions.build_regions   SLIC（动态特征 + 发射率通道）
        |                                    -> RegionGraph（无向边唯一存储）
        v
NodeSeries（置信度加权的区域均值轨迹）
        |
        +-- train_thermal.fit_thermal_operator   a_ij, beta_i, s_idle
        |
        +-- train_source.fit_source              s_i(t)（热算子冻结）
        |
        +-- uncertainty / evaluate / visualize / report
```

---

## 5. 关键实现选择

### 5.1 对称性与守恒是结构性的

每条无向边只存一个速率，两端等值反号地施加，因此 `a_ij == a_ji` **精确成立**，
训练过程中不可能漂移；耦合项写成边上的差分，对 `sum_i theta_i` 的贡献恒为零，
即网络只搬运热量、不创造热量。非负性由 `softplus` 在每一步保证。

### 5.2 速率的初始化尺度取自数据

`softplus` 在拐点以下几乎是平的，初值差一个数量级的速率头需要极多 epoch 才能
爬回来，早停会在初始化附近就触发。因此 `rate_scale` 直接取
**板级平均降温曲线的 1/e 速率**，速率头初始化在 `softplus^-1(1)`，
即从正确的数量级开始拟合。这不是结果，只是尺度。

### 5.3 划分是连续时间块，且带保护带

6 s 采样间隔下相邻帧几乎是同一次测量，随机打乱会得到"内插误差"而非泛化误差。
默认 `mode: interleaved`：把区间切成 5 个连续块，整块分配给 train/val/test，
并在每个边界丢弃 `guard_frames: 3` 帧，避免跨缝相关性泄漏。
同时报告 `mode: sequential`（前 60%/20%/20%）下的 `long_rollout_drift_k`，
那是更严格的向前外推读数。

### 5.4 先冷却、后热源

通电阶段热源与热算子存在权衡（更热可以解释为功率更大或散热更差）。
先在断电候选区间辨识 `a_ij`、`beta_i`（此处源为常数，权衡消失），
再冻结热算子拟合 `s_i(t)`。这样源被要求解释的每一开尔文，
都是传导与冷却解释不了的。

---

## 6. 测试

```bash
python -m pytest tests/ -q
```

| 文件 | 覆盖 |
|---|---|
| `tests/test_dynamic_physics.py` | 边对称性、热量守恒、非负性、无源稳定性、RK4 精度、不等间隔时间戳、批量积分、全局尺度不可辨识性 |
| `tests/test_synthetic_recovery.py` | 已知合成系统上的强耦合边恢复、冷却率结构恢复、热源定位、源不弥散、真断电时残余源≈0、均匀基线对比 |
| `tests/test_data_contract.py` | 配置解析与未知字段拒绝、板角度/窗口几何、阶段检测与再升温保护、区域图有效性、置信度加权聚合、派生量单位、真实序列数据契约 |

合成测试**不把 ground truth 交给模型**：`make_case` 分别返回观测与真值，
只有断言访问真值。

---

## 7. 产物清单

`outputs/dynamic_inverse/<sequence>/<tag>/`：

```text
temperature_corrected_sequence.npz   修复后温度序列 + mask + 区域标签 + 阶段标签
data_quality.json                    时间戳/缺帧/NaN/漂移/发射率修复统计
phase_detection.png                  阶段曲线（背景色标出各阶段）
region_partition.png                 分割 + 发射率 + 置信度
emissivity_repair.png                修复前后与提升量
thermal_coupling_graph.png           有效热耦合网络
effective_diffusivity_proxy.png      有效热扩散率 proxy
cooling_rate_map.png                 有效冷却速率
time_constant_map.png                有效时间常数
fin_length_map.png                   热扩散特征长度
held_out_rollout.png                 留出块自由 rollout 对比
normalized_source_peak.png           归一化热源峰值
normalized_source_animation.gif      归一化热源动画（+ _montage.png 关键帧）
normalized_source.npz                热源数值结果
relative_component_power.csv         相对热源份额（每列带单位）
uncertainty_maps.png                 均值/标准差/变异系数
thermal_operator.pt / source_decoder.pt   checkpoint
metrics.json / evaluation.json       指标与基线/消融
run_metadata.json                    git commit、配置、随机种子、输入摘要、单位
report.md                            自动汇总报告
```
