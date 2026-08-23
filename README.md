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
tdp/dynamic_inverse/     动态热反演（本次新增）
configs/                 measurements_v3.yaml（测量元数据）
                         dynamic_inverse.yaml（分析参数）
tests/                   物理不变量、合成数据恢复、数据契约
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
```

产物写入 `outputs/dynamic_inverse/<sequence>/latest/`，
包含 `report.md`、`metrics.json`、`run_metadata.json` 与全部图件。

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
