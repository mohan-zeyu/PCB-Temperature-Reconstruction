# PCB 红外热成像：真实温度场重建与热演化预测

红外热像仪会把低发射率的金属读得比真实温度低：SoC 顶盖、连接器、屏蔽罩
都是这样。本项目用同一块板的两组记录重建真实表面温度，并预测板的温度演化。

两条入口：

1. **Thermal Studio（网页）**：任意一块板，上传裸板与高发射率（黑化）记录，
   服务端完成定位、透视校正、结构配准、不可信区掩膜、按热状态配对，再训练
   这块板自己的残差校正模型。之后任意一帧同板 CSV 或原始热像 JPEG 都能重建。
2. **case10 数值基准（桌面）**：树莓派 4B 满载，240 个通电帧的逐帧重建，
   以及从任意帧开始的 674 秒、一秒间隔前向预测。

**发射率不拟合。** 相机的 0.95 设定定义可信像素的真值；低发射率区域用同板的
高发射率记录做配对校正。从温度时间序列里反推发射率噪声太大，也分不清发射率
和温度，所以不这么做。

**只有相对温度。** 热容、热导、换热和功率被同一个未知常数缩放，温度轨迹完全
不变。仓库不输出瓦特，也不声称恢复铜层几何。详见「单位与命名」。

## 快速开始

需要 Python 3.12。

```bash
uv sync                     # 或 python -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python -m pytest tests/ -q          # 126 个测试
.venv/bin/python web/server.py --port 8000    # 网页工作台
# 打开 http://127.0.0.1:8000/web/
```

两个内置板型随仓库分发在 `models/boards/`，首次启动自动注册到
`outputs/web_boards/`。板型库可以导出权重 zip，也可以导入别人训练的板型；
训练或导入的板型同样保存在 `outputs/web_boards/`（该目录不进 git）。

## 网页工作台

「训练新板」需要同一块板的两组记录：

| 组 | 作用 |
|---|---|
| 裸板 | 低发射率金属读低的原始记录，模型输入 |
| 高发射率（黑化 / 黑胶带） | 同一块板的可信表面温度，模型标签 |

服务端顺序执行：

1. 解析 CSV 或原始热像 JPEG（任意传感器分辨率）；
2. 定位板体，按物理尺寸统一网格；
3. 把高发射率窗口按梯度结构配准到裸板窗口；
4. 在裸板上检测低发射率区域；
5. 按可信区热状态配对两段序列，不依赖帧号或日志；
6. 训练残差 U-Net；
7. 保存模型、窗口、掩膜、结构签名、训练曲线和预览。

「温度重建」上传任意一帧同板数据：

- 依次试 0°、90°、180°、270°，取结构匹配最高的方向；
- 结构匹配低于 0.45 时，用板型自己的参考重新配准窗口（平移 + 缩放）；
- 返回裸板读数、重建温度、不可信区掩膜、结构匹配分和像素读数。

板型库支持导出权重 zip（`board.json`、`model.pt`、`arrays.npz`、`preview.npz`）
和导入同样的 zip。这样不用重新训练就能分享一块板，例如 VisionFive 2。

已训练的板型：

| 板型 | 网格 | 配对 | 训练集板体 MAE |
|---|---|---:|---:|
| Raspberry Pi 4B | 157×103 | 432 | 0.199 K |
| Orange Pi 5 Pro | 164×103 | 480 | 0.170 K |
| VisionFive 2（本机训练，不随仓库分发） | 184×133 | 240 | 0.258 K |

这些是**训练集拟合误差**，不是留出泛化。

接受的输入：

- **HIKMICRO CSV**：256×192 或任意网格，0.1 K；
- **本项目采集的原始热像 JPEG**：JPEG 图像之后带 HDRI 温度块，解码后与
  CSV 相差 0.11 K（case10 全部 480 帧）；
- **相机原生的 192×256 SDMP JPEG**（带 `Radiometric Image` Exif）：温度流
  是厂商私有编码，当前不解码。用 HIKMICRO Analyzer 导出 CSV，或上传 HDRI
  格式的 JPEG。服务端会识别并给出这条提示。

## case10 数值基准

桌面流程处理 case10（树莓派 4B 裸板满载）与 case17（同板哑光黑漆满载）。
case17 提供可信表面温度，两次实验的 SoC 日志提供结温锚点。**case17 是标定
输入，不是独立验证集。**

**Stage 2：逐帧重建。** 240 个通电帧全部重建。被动块用外圈拟合的源解，SoC
用一个正宽源在同一次平板求解里生成。case17 稳态表面在 SoC 核心 78.28 °C、
峰值 79.25 °C，结温 83.04 °C；把 4.76 K 的结温到表面差移到 case10，得到
稳态表面核心 77.78 °C、峰值 **78.29 °C**。同一掩膜下，无锚点峰值是
69.02 °C，旧纯图像流程是 68.97 °C。每个像素同时保存模型范围。

**Stage 3：前向预测。** 固定模式库有 64 个慢平板模式和多组源响应；闭式矩阵
解支持任意正时间间隔和直接稳态。冻结的连续划分（带保护带）上：

| 模型 | 留出测试加权 MAE |
|---|---:|
| 本期生产模型（向训练段稳态衰减） | **0.192 K** |
| 保持不变 baseline | 0.271 K |
| 常数源 | 1.574 K |
| 少量平滑正源（失败消融） | 2.032 K |
| 无源 | 15.178 K |

从第 0 帧起覆盖整个通电段（674 个一秒场）的加权 MAE 是 0.427 K。起始帧
严格不跳变。方法与消融见
[stage3_forecasting.md](docs/plans/stage3_forecasting.md)。

```bash
.venv/bin/python scripts/run_reconstruct.py      # Stage 2
.venv/bin/python scripts/run_patterns.py         # Stage 3 模式库
.venv/bin/python scripts/run_forecast_fit.py     # 训练与冻结评估
.venv/bin/python scripts/run_forecast.py --start-frame 0
.venv/bin/python scripts/figure_forecast.py      # 1280×720 视频与消融图
```

产物在 `outputs/thermal/case10/`。

## 数据

`实验数据（黑胶带版）/` 是采集原始数据，约 4.8 GB，不进 git。

```text
caseNN_.../data/HM<YYYYMMDDHHMMSS>.csv   256×192 摄氏度矩阵
caseNN_.../photo/HM<...>.jpeg            同一帧的伪彩图 + HDRI 温度块
记录csv文件/                              SoC、CPU 负载与环境日志
```

树莓派 4B：

| 序列 | 表面 | 负载 | 帧数 |
|---|---|---|---:|
| case09 | 裸板 | 50% | 480 |
| case10 | 裸板 | 100% | 480 |
| case16 | 哑光黑漆 | 50% | 324 |
| case17 | 哑光黑漆 | 100% | 293 |

Orange Pi 5 Pro：case11–case15（裸板与黑胶带，0/50/100% 负载）。

VisionFive 2（2026-11-07）：裸板与黑漆板各 6 组，满载/半载各 240 帧、
空载 60 帧、无电源 30 帧，以及两段冷却各 240 帧，间隔 6.3 秒。解压目录：

```text
实验数据（黑胶带版）/11.07Vision_Five_2_裸板数据/
实验数据（黑胶带版）/11.07Vision_Five_2_黑漆板数据/
```

**必须记住的事实：**

- 板子**从未断电**。负载停止后仍稳定在环境温度以上约 14 K，任何拟合都不
  假设零功率。
- 没有实测像素尺度。毫米数来自 PCB 外形（树莓派 85×56 mm）与重新采样的
  网格；`pixel_size_calibrated` 保持 `false`。
- 相机 CSV 不带辐射参数块，0.95 是**假设**，不是测量。
- case10 的配对掩膜把约 38% 的板面标为不可信，最深处离可信像素约 20 mm。

## 单位与命名

热容、热导、换热和功率被同一个未知常数缩放，温度轨迹完全不变。因此**不要
输出** `W`、`W/K`、`W/(m·K)`、`W/m²`，也不要声称恢复了铜层几何或“某个
元件占多少功率”。可以比较热点排序和相对变化。

用平实的中文名称，不用希腊字母：

| 用这个 | 不用 | 单位 |
|---|---|---|
| 扩散速率 | alpha, diffusivity | px²/s |
| 冷却速率 | beta | 1/s |
| 加热速率 | source | K/s |
| 响应时间 | tau | s |
| 温度高于环境 | theta | K |

## 项目结构

```text
tdp/io/             CSV 与 JPEG 读取，统一上传入口
tdp/data/           板体定位、透视校正、发射率区域检测
tdp/thermal/        Stage 2/3：源检测、标定、重建、模式库、预测
tdp/unet/           网页板型的残差 U-Net
tdp/web/            网页后端：上传、配对、训练、推理、注册表
tdp/dynamic_inverse/ 旧的动态反演对比路径（不参与当前结果）
web/                Thermal Studio 前端与启动脚本
models/boards/      随仓库分发的两个预置板型
scripts/            运行与出图入口
docs/plans/         Stage 2/3 方法记录、未来固定负载计划
tests/              126 个测试
```

## 边界

- **网页模型只训练一条会话。** 上报的是训练集误差；要独立验证，需要第三次
  同板、同负载、同步记录结温的会话。
- **高发射率记录是标签，不是绝对真值。** 它同样建立在 0.95 假设上。
- **假设同一取景与姿态。** 四个 90° 方向和几像素平移能处理；大角度旋转、
  明显缩放或换镜头不能。
- **未知物理尺寸时长度只有像素单位。**
- **原生 SDMP JPEG 不解码**，见上。
- **固定负载模型尚未实现。** 需要 5 个负载档、每档 3 次独立实验（共 15 次；
  若 case09/case10 合格可计入，则至少再补 13 次）。数据门槛与验收标准见
  [future_fixed_load_forecasting.md](docs/plans/future_fixed_load_forecasting.md)。

## 文档

- [CLAUDE.md](CLAUDE.md)：当前决策、约束与已实现状态，冲突时以它为准。
- [stage2_reconstruction.md](docs/plans/stage2_reconstruction.md)：重建方法。
- [stage3_forecasting.md](docs/plans/stage3_forecasting.md)：预测方法与冻结结果。
- [web/README.md](web/README.md)：网页工作台、API 与训练细节。
- [future_fixed_load_forecasting.md](docs/plans/future_fixed_load_forecasting.md)：
  固定负载计划。
- [docs/dynamic_inverse_zh.md](docs/dynamic_inverse_zh.md)：旧动态反演说明。
