# GapSet-R

Relative support correction for incomplete CSI gesture recognition.

本仓库对应论文 **GapSet-R：不完整CSI手势识别的相对支持校正**。模型将可见锚点、跨度校准和支持加权空间汇总连接起来，再通过特征 H、实际支持 R 与参考支持 P 的相对读出保留参考累计尺度。参考支路不重建隐藏 CSI，也不增加独立参数。

## 仓库内容

|路径|用途|
|---|---|
|`src/gapset/models/relative_model.py`|完整 GapSet-R，73,100 个参数|
|`src/gapset/models/`|论文基线、最终结构匹配消融、绝对归一化路径下的结构对照|
|`src/gapset/data/`|Widar 幅度预处理、组划分、common8 编码与固定缺失协议|
|`src/gapset/training/`|训练、源域检查点选择、评估、数值门禁及资源保护|
|`src/gapset/analysis/`、`tools/`|论文的组级分析、支持诊断、内容干预、线索分类、FLOPs 与结果绘图|
|`checkpoints/`|仅完整 GapSet-R 的三个论文检查点，种子42/43/44|
|`results/`|最终稿表1–6（含表4a/b/c）的 CSV 和图3/4的既有汇总数值|
|`tests/`|不读取真实数据、不做模型前向的协议测试|
|`environment.yml`、`requirements-cuda.txt`|Conda环境及按归档版本安装的CUDA扩展|

仓库不包含其他模型权重、原始 CSI、逐窗口预测、实验机配置、实验日志、论文 Word/PDF、旧版图和写作过程文档。基线代码用于复现共同协议比较，不代表重现各原论文完整系统。MIT 许可和必要的第三方声明见 [LICENSE](LICENSE)、[THIRD_PARTY.md](THIRD_PARTY.md)。

## 环境

归档正式实验记录的关键版本为 **Python 3.11、PyTorch 2.5.1+cu124、NumPy 2.4.6、Mamba 2.2.6.post3、causal-conv1d 1.4.0、Triton 3.1.0、transformers 4.44.2**。先前发布说明中的2.7.1等版本已按实际记录更正。计算使用FP32并关闭TF32。

从仓库根目录创建Linux/WSL + NVIDIA环境：

```bash
conda env create -f environment.yml
conda activate gapset-r
python -m pip install --no-build-isolation -r requirements-cuda.txt
python -m pip check
python -m gapset --help
python tools/check_weights.py
python -m unittest discover -s tests -v
```

`environment.yml`先安装PyTorch及项目的原始数据、基线和绘图依赖；`requirements-cuda.txt`再安装CUDA扩展，以便构建时使用已经安装的PyTorch。CUDA扩展需要适配的预编译wheel，或与PyTorch匹配的CUDA开发工具链（含`nvcc`）；PyTorch wheel自带的CUDA运行库不等于完整编译工具链。GPU训练与正式评估使用Linux/WSL，文件锁使用`fcntl`。

该YAML固定了归档中已记录的关键科学依赖，并非整个操作系统及全部传递依赖的逐字节锁文件；本次没有新建或安装环境。CUDA扩展安装要求可参阅[Mamba上游说明](https://github.com/state-spaces/mamba/tree/v2.2.6.post3)，PyTorch构建见[官方历史版本](https://pytorch.org/get-started/previous-versions/#v251)。CPU参考递推及权重检查不需要Mamba CUDA扩展。

部分基线作者源码未在固定版本中找到明确再分发许可，因而仅提供版本锁和校验下载入口，不随仓库发布。详见 `THIRD_PARTY.md`：

```bash
python tools/fetch_baselines.py --component all
```

## 使用论文权重

`checkpoints/manifest.json` 记录种子、选中轮次、源检查点 SHA-256 和发布文件 SHA-256。三个检查点分别来自第176、163、172轮，均按干净源域验证宏F1首次严格最大值选择，没有按挑战成绩挑选种子。发布文件保留完全相同的模型张量及最小身份信息，去除优化器、随机状态、历史记录和私有路径。权重获取状态以清单为准，缺失文件时加载器会明确报错。

```bash
python tools/check_weights.py
```

```python
from gapset.checkpoints import load_paper_model

model, metadata = load_paper_model(seed=42, directory="checkpoints")
# model(image, timestamp): image[B,8,30,256], timestamp[B,256]
# image必须由common8接口按fit-only统计编码；不能直接传未处理的CSI热图。
```

三个种子的干净宏F1为 **73.19±0.15%**，45项缺失挑战平均宏F1为 **58.96±0.49%**（样本标准差）。这是 Widar 六类跨域开发协议结果，不是未接触的独立最终测试。各种缺失类型、人员和类别的结果均保留；掩码/时长线索与开发集参与设计反馈的适用范围见论文。FLOPs 是规定口径下的算法估计，不是实测时延。

## 数据准备

按 [Widar3.0 数据集作者要求](https://tns.thss.tsinghua.edu.cn/widar3.0/)获取原始 `.dat`，将日期目录放入 `data/raw/CSI`。不随代码分发数据。

```bash
python -m gapset prepare --raw data/raw/CSI --output data/cache --workers 2
python -m gapset audit --data data/cache --raw-root data/raw/CSI --output outputs/input_identity.json
python -m gapset freeze-masks --data data/cache --output outputs/expected_masks.json
```

输入为3条天线流、30个原生子载波、256个10ms时间箱。幅度在可见值上做 `log1p` 后用训练集统计标准化；掩码、时间位置与子载波位置组成 common8 输入。所有 RX 和窗口按原始手势组共同划分。固定45项挑战覆盖 RF、Span、Step、Burst、Link、Joint；Span 保留连续子载波区间，图中横轴表示被删除比例。

## 评估论文检查点

下面是用户自行运行的新评估入口；仓库整理不会执行评估。默认覆盖源域验证与跨域开发的各46项条件，不重新训练或选模：

```bash
python tools/evaluate_checkpoint.py --seed 42 --data data/cache --input-identity outputs/input_identity.json --expected-masks outputs/expected_masks.json --output outputs/paper_s42 --max-hours 12
```

分别使用42/43/44复现三种子结果。源检查点清单中的 manifest 哈希标识原实验缓存；公开预处理后的本地 manifest 因原始文件路径不同可有不同哈希，评估入口会核对本地输入审核与冻结掩码是否匹配同一个缓存。

表6的固定权重干预使用种子42与干净输入：

```bash
python tools/evaluate_checkpoint.py --seed 42 --data data/cache --input-identity outputs/input_identity.json --expected-masks outputs/expected_masks.json --split cross_domain_dev --conditions clean --intervention fit_mean_content --output outputs/content_mean
python tools/evaluate_checkpoint.py --seed 42 --data data/cache --input-identity outputs/input_identity.json --expected-masks outputs/expected_masks.json --split cross_domain_dev --conditions clean --intervention mask_pattern_time_shuffle42 --output outputs/content_shuffle
python tools/cue_controls.py --data data/cache --output outputs/cue_controls
```

## 从头训练及消融

训练使用干净输入200轮，物理batch128、累积4步、有效batch512；AdamW、学习率1e-4、权重衰减1e-3、余弦200轮、CE、梯度裁剪1，无人工缺失增强、外部预训练或早停。配置见 `configs/paper_protocol.json`。先对一个案例运行CPU与CUDA门禁，再训练、评估：

```bash
python -m gapset cpu-gate --model ours_gapset_relative_mass_control --seed 42 --data data/cache --input-identity outputs/input_identity.json --output outputs/relative_s42/cpu
python -m gapset gate --model ours_gapset_relative_mass_control --seed 42 --data data/cache --input-identity outputs/input_identity.json --cpu-gate outputs/relative_s42/cpu/cpu_gate.json --output outputs/relative_s42/gate --max-hours 2
python -m gapset train --model ours_gapset_relative_mass_control --seed 42 --data data/cache --input-identity outputs/input_identity.json --cpu-gate outputs/relative_s42/cpu/cpu_gate.json --output outputs/relative_s42/train --max-hours 24
python -m gapset eval --model ours_gapset_relative_mass_control --seed 42 --data data/cache --input-identity outputs/input_identity.json --cpu-gate outputs/relative_s42/cpu/cpu_gate.json --training outputs/relative_s42/train --expected-masks outputs/expected_masks.json --output outputs/relative_s42/eval --max-hours 12
```

|论文对照|`--model`|
|---|---|
|完整GapSet-R|`ours_gapset_relative_mass_control`|
|取消质量校正|`ours_gapset_state_no_mass_norm_control`|
|最终结构固定间隔差分|`ours_gapset_relative_fixed_lag`|
|绝对归一化变体|`absolute_norm`|
|表4c四个补充结构对照|`early_fixed_lag`、`ordered_antennas`、`ordered_normalized`、`grid_state`|
|GRU-D common8适配|`grud_common8`|

表4c的后四行均以绝对归一化路径为参照，不能当作最终模型的独立模块消融。GRU-D的训练前准备：

```bash
python -m gapset fit-means --data data/cache --output third_party/fit_means.json
```

每个训练案例使用新目录。资源保护保留单GPU锁、3GiB预留、82°C退出阈值、80GiB磁盘底线；遇到其他计算任务退出自身，OOM停止，不改batch。本地入口不含远程调度、自动搜索或旧任务恢复。

## 重画结果与分析

```bash
python tools/plot_paper.py --output generated_figures
python tools/complexity.py --model gapset_relative --output outputs/flops.json
```

第一条只读取已包含的汇总数值，输出图3六模型三种子热图、图4四类缺失曲线的600dpi PNG和SVG。SD按三个训练种子计算；Burst曲线先在种子内平均三个位置，热图则保留位置。Times New Roman不可用时回退DejaVu Serif。论文中的真实CSI示例与作者排版的结构图不在此仓库分发原始数据或PPT。

`tools/build_group_counts.py`将完整评估产物按原始手势组汇总；`tools/analyze_groups.py`执行固定人员内、501次配对组重采样。前者参数为`--manifest`、`--evaluations`、`--output`，后者为`--input`、`--output`，见各入口`--help`。`tools/support_diagnostics.py`接受`configs/support_diagnostics.example.json`格式的新计划，仅用于种子42、原预定256窗口支持诊断；需自行填写数据路径、权重SHA及新的运行截止。

## 代码整理核验

本次检查包括源码/许可清单、核心计算AST一致性、论文表格与既有数值核对、协议测试、CLI入口、结果图重画，以及权重到正式评估哈希的对应关系。完整检查结果见仓库外的交付说明。整理后没有重新完成200轮训练或真实数据评估，也没有新增硬件效率结果。
