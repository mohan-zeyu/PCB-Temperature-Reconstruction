# PCB 红外热成像：发射率修复与动态热反演

从红外热像序列反演印刷电路板的**可辨识**热参数。当前对象：树莓派 4B。

本仓库刻意区分两类结果：

* **可辨识**——温度轨迹本身能确定的量：有效热耦合速率、冷却速率、
  时间常数、有效热扩散率 proxy、热扩散特征长度、归一化热源速率、相对热源份额；
* **不可辨识**——温度轨迹原理上无法确定的量：材料热导率 `W/(m·K)`、
  绝对面热导 `W/K`、绝对功率密度 `W/m²`、元器件功率 `W`。

原因是全局尺度不确定性：把面热容 `C`、面热导 `G`、换热系数 `h` 与功率 `q`
同乘一个常数，所有温度轨迹完全不变。**本仓库不输出、不声称第二类量。**
详见 [`docs/dynamic_inverse_zh.md`](docs/dynamic_inverse_zh.md) 第 1 节。

## 目录

```text
tdp/io/                  HIKMICRO CSV 解析（自带统计量自检）
tdp/data/crop.py         板窗口定位、旋转/透视矫正、配准
tdp/data/emissivity.py   低发射率区域检测与传导修复
tdp/thermal/             热源检测、逐帧重建、固定物理模式与闭式前向求解
tdp/dynamic_inverse/     动态热反演
tdp/web/                 网页后端：任意板体上传、配对、训练、推理
web/                     Thermal Studio 前端、内置样例与启动脚本
configs/                 measurements_v3.yaml（测量元数据）
                         dynamic_inverse.yaml（分析参数）
tests/                   物理不变量、合成数据恢复、数据契约、网页后端
docs/                    中文说明
实验数据（黑胶带版）/     原始采集数据
```

## 安装

```bash
uv venv --python 3.12
uv pip install numpy scipy torch matplotlib pandas pyyaml scikit-image pillow pytest imageio
```

## 快速开始

```bash
# CPU 冒烟测试（约 15 秒跑通全链路）
python -m tdp.dynamic_inverse run --smoke --sequence case10_rpi4b_bare_full

# 完整流程
python -m tdp.dynamic_inverse run --sequence case10_rpi4b_bare_full

# 测试
python -m pytest tests/ -q

# case10 热源检测与 240 个通电帧的最终重建
python scripts/run_reconstruct.py

# Stage 3a：固定物理模式库与闭式平板求解器
python scripts/run_patterns.py

# Stage 3b：冻结数据划分、拟合并验证预测模型
python scripts/run_forecast_fit.py

# 从任意已重建帧开始，按 1 秒间隔正向预测至稳态（默认第 0 帧）
python scripts/run_forecast.py --start-frame 0

# 固定 1280x720 视频、误差曲线与冻结测试集消融图
python scripts/figure_forecast.py

# 网页工作台：任意板子上传训练，任意同板 CSV 真实推理
python web/server.py --port 8000
# 打开 http://127.0.0.1:8000/web/
```

产物写入 `outputs/dynamic_inverse/<sequence>/latest/`，
包含 `report.md`、`metrics.json`、`run_metadata.json` 与全部图件。

## 网页工作台：任意板子、任意原始数据

`web/server.py` 把原来的静态演示变成真实服务：

* 「训练新板」上传同一块板的裸板与高发射率（黑化）CSV，服务端会定位板体、
  统一网格、按结构把两组画面配准、检测不可信区域、按热状态配对，再训练该板
  自己的残差校正模型；
* 「温度重建」上传任意一帧同板 CSV，服务端用该板型的窗口、掩膜和权重做真实
  推理，不再要求文件哈希与内置样例一致；
* 每个新板型保存在 `outputs/web_boards/<board-id>/`，可删除、可切换；
* 没有后端时页面进入只读演示模式，只显示内置预测，不会假装训练。

**边界**：新增板型只训练一条会话，上报的是训练集误差，不是留出泛化；高发射率
记录是标签而不是绝对真值；未知板体尺寸时长度只有像素单位。详细说明见
[`web/README.md`](web/README.md)。

最终的热源驱动重建写入 `outputs/thermal/case10/`。默认流程把 case10 与同负载、
哑光黑漆的 case17 按绝对时间和负载区间对应，并用两次实验的 SoC 日志把结温修正为
可见表面温度；旧的纯图像流程可用 `--no-paired-calibration` 复现。SoC 稳态表面峰值
由旧流程末帧 68.97 °C 修正为 78.29 °C（case17 表面峰值 79.25 °C）。可信区也不再
由“看起来热”来保证：配对衰减审核把 43 个遗漏像素补入不可信掩膜，但不拟合发射率。
SoC 使用宽分布正热源和同一个平板方程生成平滑平台；输出同时保存标定输入、最终重建
源、外圈拟合源、完整重建掩膜和逐像素模型范围。方法与局限见
[`docs/plans/stage2_reconstruction.md`](docs/plans/stage2_reconstruction.md)。

Stage 3a 从上述重建归档读取几何、热源和不确定度，生成 64 个慢平板模式及多组
热源响应模式，并用闭式矩阵解支持任意正时间间隔和直接稳态求解。Stage 3b 冻结
连续训练/验证/测试区间，并实际检验了“少量平滑正热源”方案；该方案在留出测试上
为 2.032 K，差于 0.271 K 的保持不变基线，因此只作为失败消融保留。生产模型使用
训练区间末段（147--188 帧）估计完整稳态场，再让任意起始帧相对该稳态的偏差按稳定
平板算子衰减。留出测试加权 MAE 为 0.192 K，起始帧严格不跳变，且 SoC 的宽平台
不会退回无热源形状。环境温度采用“保持起始值”的外部情景，不由模型预测；逐像素
范围是模型范围而非统计置信区间。方法见
[`docs/plans/stage3_forecasting.md`](docs/plans/stage3_forecasting.md)。

Stage 3c 不再拟合模型，只读取冻结产物，生成包含全部 674 个一秒场的 1280×720
H.264 视频、四关键帧 montage、逐帧误差数组以及留出集消融图。若系统没有 MP4
编码器则回退为抽样 GIF；montage 和数值数组始终保留。

下一阶段计划把每次实验中固定不变的负载浮点数（0 到 1）作为额外模型输入。该模型
尚未实现；在至少取得 5 个负载档位、每档 3 次独立实验（共 15 次）之前，当前 case10
模型保持不变。完整的数据门槛、划分方式和模型约束见
[`docs/plans/future_fixed_load_forecasting.md`](docs/plans/future_fixed_load_forecasting.md)。

## 数据说明

`实验数据（黑胶带版）/case*/data/HM<YYYYMMDDHHMMSS>.csv` 为 256×192 摄氏度矩阵；
同名 `photo/*.jpeg` 是**相机对同一帧热数据的伪彩渲染**，不是可见光照片。

树莓派 4B 相关序列：

| 序列 | 表面 | 负载 | 帧数 | 时长 |
|---|---|---|---|---|
| `case09_rpi4b_bare_half` | 裸板 | 50% | 480 | 53 min |
| `case10_rpi4b_bare_full` | 裸板 | 100% | 480 | 50 min |
| `case16_rpi4b_painted_half` | 哑光黑漆 | 50% | 324 | 38 min |
| `case17_rpi4b_painted_full` | 哑光黑漆 | 100% | 293 | 31 min |

每条序列都包含完整的「升温 → 准稳态 → 卸载降温」循环。

**重要**：`记录csv文件/soc_log_100.csv` 显示卸载后 SoC 仍稳定在 46–48 °C，
即**负载停止但板子从未断电**。因此降温区间在所有输出中标记为
`free_cooling_candidate`，拟合时不假设 `q=0`，而是把残余空载源作为
常数非负量一并辨识。
