# Week 8：自回归模型、尺度规律与顿悟

HUANG QIUSHAN · AMAT5315 2026 Fall。四个核心部分已在本机 CPU 上训练、评估和整理；
另以本机完成可选挑战的宽度 256、384 计算，未租用云机器。先看
[中文审核页](review.html)、[学习笔记](STUDY_NOTES.zh.md)和[逐项复核](STRICT_AUDIT.zh.md)。
课程原文要求 Week 9 开始时展示，并写明没有另行上传提交；原文要求推送源码、环境
和证据。

## 资料与实现来源

原始学习单 PDF、资源 ZIP、解压文件、语料与参考文献位于本机课程目录的
`../../../week8/`；线上入口是[学习单](https://giggleliu.github.io/AMAT5315-2026Fall/week8-learning-sheet.html)
与[课程资源包](https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week8-resources.zip)。
学习单 PDF 的 SHA-256 为 `4a9f5d0cc5223ed6c1863410cc1f4e03b890dba21649265ca8a91eacd0b95101`，
资源 ZIP 为 `5fa9290eafc9f8432a99c6af21748e6371c0b3676b9e1c357e46d3ed9cd287f5`。

`cpu/`、`grokking/`、`speedrun/`、`checker/` 和 `data/*figures.py` 从课程资源包的
教师参考代码复制并调整；这些代码并非声称从零独立编写。我们按学习单将生成长度
从 300 改为 500；随机标签改为真实标签的排列；记录训练开始时间并阻止覆盖已冻结
的预测；修复独立评估的相对路径和可选扩展的协议选择；补充所有种子的绘图、测试、
验证脚本和本页说明。课程包中的 `reference.json`、参考模型输出和参考图片**没有**
作为本次训练结果。`checker/evaluator.py` 对导出权重独立执行 NumPy 前向与评分。
课程原始校准数据和未使用的 fixture／对照脚本留在本机资料中，不列入提交。

## 本次结果

TinyShakespeare SHA-256：
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`。
训练/验证/计分字符数分别为 1,003,854 / 111,540 / 111,539。所有分数均按 128 字符
的块重置上下文并按 token 数加权；最后一个块只有 51 个目标。

| 预测器 | 验证损失，nats/字符 | 困惑度 |
| --- | ---: | ---: |
| 均匀 | 4.174387 | 65.000 |
| 一元 | 3.347305 | 28.426 |
| 二元 | 2.481889 | 11.964 |
| 两层 Transformer | 1.771941 | 5.882 |

训练、验证与计分字符数，以及均匀和两目标示例的损失与困惑度，见
[`evidence/baselines.txt`](evidence/baselines.txt)；由 `uv run python cpu/baselines.py` 打印。

Transformer 为宽度 64、4 头、116,673 个参数；3,000 步，batch 16，seed 1337。
八组因果扰动中前 64 个位置的最大 logits 变化为 0；第一层注意力上三角最大值为 0，
行和最大误差为 1.19×10⁻⁷。保存权重的 JAX 分数与独立 NumPy 分数相差 6.7×10⁻⁹。
温度 0.5 和 1.0 各生成了 500 个新字符，完整文本在
[`evidence/cpu-measurements.json`](evidence/cpu-measurements.json) 的 `part2.samples`。

尺度实验用宽度 12–128 的八个模型、每个两个种子，宽度 8 排除；所有实验使用 1,500
步和相同 3,072,000 token 预算。按非嵌入参数数拟合的 α=0.07170±0.00259，Birge 比率
3.126。时间均为北京时间 2026-10-28：

| 宽度 | 预测及 95% 区间 | 实测两种子均值 | 预测保存 | 最早训练开始 | 首次运行完成 | 命中区间 |
| ---: | --- | ---: | --- | --- | --- | --- |
| 192 | 1.6352 [1.5808, 1.6922] | 1.6556 | 11:59:32 | 11:59:39 | 12:00:35 | 是 |
| 256 | 1.5692 [1.5132, 1.6278] | 1.6307 | 12:02:42 | 12:03:40 | 12:05:36 | 否 |
| 384 | 1.4807 [1.4227, 1.5410] | 1.6856 | 12:03:15 | 12:03:40 | 12:06:28 | 否 |

256 与 384 的预测均在**两种宽度的任何一次训练开始之前**保存；没有把新结果加入
拟合。后两次失准是本次结果，不是验证代码错误；固定训练预算下可能存在优化或数据
限制，但目前没有额外实验足以归因。教师学习单在开始之前已经公开其参考结果，因此
这些记录证明本次执行顺序，不能声称对已知参考答案实现完全盲法。

课程参考实现使用按误差传播加权的**对数损失空间**线性拟合；PDF 中展示的 χ² 则
写在原始损失空间。独立的事后计算直接最小化后一目标，宽度 192 的中心预测为
1.63467，和原事前预测 1.63516 相差 −0.00049。这个事后比较只用于说明近似误差，
不替换已冻结的预测或区间。

模加法五个主种子都在第 200 步达到训练准确率 0.99，测试达到 0.99 的步数分别为
6000、5800、7600、4500、3400，延迟比为 30、29、38、22.5、17。无权重衰减的三个
对照在 10,000 步末测试准确率为 0.144、0.117、0.041；50% 训练对的实验在第
300–400 步达到 0.99。打乱标签的实验在第 200 步记住训练集，末态测试准确率
0.01093，与机会水平 1/97≈0.01031 接近。

五个主模型最终的前五个频率占非恒定嵌入功率 0.886–0.955。seed 0 的关键频率为
2、18、32、33、38：只保留它们，测试准确率 0.99954；去掉它们，准确率 0.00926；
去掉十组随机非关键频率均为 1.0，只保留一组随机非关键频率为 0.00926。
[`grokking/numpy_check.py`](grokking/numpy_check.py) 独立复算了 seed 0 的各阶段
准确率、傅里叶功率、关键频率和消融。

## 图与生成命令

以下命令在 `week8/` 执行，要求相应的原始本机记录已存在。绘图只读取记录，不训练。

| 图 | 产生它的命令 | 内容 |
| --- | --- | --- |
| [cpu-part1-probabilities.png](evidence/cpu-part1-probabilities.png)、[cpu-part1-baselines.png](evidence/cpu-part1-baselines.png) | `uv run python data/cpu-figures.py` | 训练集字符概率与验证损失/困惑度 |
| [cpu-part2-loss.png](evidence/cpu-part2-loss.png)、[cpu-part2-attention.png](evidence/cpu-part2-attention.png) | `uv run python data/cpu-figures.py` | 训练曲线、四个头的因果掩码 |
| [cpu-part3-scaling.png](evidence/cpu-part3-scaling.png)、[cpu-part3-scaling-extra.png](evidence/cpu-part3-scaling-extra.png) | `uv run python data/cpu-figures.py` | 192 的事前预测和 256/384 外推反例 |
| [grokking-curves.png](evidence/grokking-curves.png)、[grokking-controls.png](evidence/grokking-controls.png) | `uv run python data/grokking-figures.py` | 全部主种子的训练/测试和对照 |
| [grokking-fourier.png](evidence/grokking-fourier.png)、[grokking-ablation.png](evidence/grokking-ablation.png) | `uv run python data/grokking-figures.py` | 频率功率与消融 |

## 环境与复现

使用 macOS Apple M4 Max、CPU JAX；Python 3.14.7，JAX/jaxlib 0.10.2、Optax 0.2.8、
NumPy 2.4.6、SciPy 1.18.1、Matplotlib 3.11.2。依赖及解析版本见
[`pyproject.toml`](pyproject.toml) 与 [`uv.lock`](uv.lock)。本机 uv 可执行文件是
`../week7/.bootstrap/bin/uv`；下方以 PATH 中的 `uv` 表示。

在新的空运行目录中，先 `uv sync --locked`，将官方原始 `input.txt` 放到 `data/input.txt`
并复制到 `speedrun/input.txt`；必须检查上面的 SHA-256。原始语料、模型、优化中间状态
和原始运行均由 `.gitignore` 排除。尺度和顿悟的原始运行分别在 `artifacts/scaling/`
和 `artifacts/grokking-runs/`；为保留原始现场，旧路径的字节一致副本也留在本机并被
忽略。以下是运行顺序；已有结果的目录不要用于重新开始
一个新实验，预测脚本会拒绝覆盖已有预测或在目标模型已开始后伪称盲测。

```sh
uv run python cpu/baselines.py
uv run python cpu/part2.py cpu/configs/part2.json --out artifacts/part2
uv run python cpu/scaling.py train-fit cpu/configs/scaling-b.json --jobs 4
uv run python cpu/scaling.py predict cpu/configs/scaling-b.json
uv run python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('data/cpu-measurements.json').read_text())['part3b']['prediction']
Path('artifacts').mkdir(exist_ok=True)
with Path('artifacts/scaling-prediction.json').open('x') as f:
    json.dump(p, f, indent=2)
PY
uv run python cpu/scaling.py train-held cpu/configs/scaling-b.json --jobs 2
uv run python cpu/scaling.py compare cpu/configs/scaling-b.json
uv run python run_grokking_jobs.py --jobs 3
uv run python grokking/run_study.py --only main wd0 frac05 random
uv run python data/cpu-figures.py
uv run python data/grokking-figures.py
```

可选宽度 256、384 必须在两者训练前各执行 `predict-extra` 并保存预测快照，再分别
执行 `train-extra`、`compare-extra`；[云端挑战说明](CLOUD_CHALLENGE.zh.md)列出完整命令
及费用假设。`run_grokking_jobs.py` 对每个实验单独保留日志并缓存结果，不会将参考数据
当作本次计算。

第一次 Part 2 训练完成并保存权重后，评分子进程因相对路径失败；原始训练日志和权重
保留，路径已修复，使用 `--recover-log artifacts/logs/part2.log` 从同一权重复算完整
验证分数、因果探针和文本。曲线从当时控制台恢复，训练和局部验证损失保留四位小数，
不是声称恢复了丢失的完整精度；最终分数是从权重重新计算的。

## 核验

```sh
uv run python -m pytest tests cpu/test_cpu.py grokking/test_grok.py checker/test_checker.py -q
uv run python grokking/numpy_check.py
uv run python verify.py
uv run python strict_audit.py
```

本次自动测试结果见 [tests.txt](evidence/tests.txt)；`verify.py` 的 59 项核验和
`strict_audit.py` 的 306 项严格复算通过，记录见 [verification.json](evidence/verification.json)
与 [strict-audit.json](evidence/strict-audit.json)。这里的“通过”不把 256/384 的区间
失准视为代码失败。在只复制待提交文件和官方语料的干净目录中，离线重建锁定环境、
运行 35 项测试及 2 个子测试，并完成字符模型训练/导出/独立评分和模加法模型的小规模
训练检查；这个小规模检查不冒充完整实验的复跑。尺度扫描四个进程并行，JAX 设置了单线程 Eigen 标志，但 macOS 没有
对每个进程强制单核亲和性，运行期间观察到多核使用；这一资源执行细节不同于学习单
建议的“一进程一核”。课程包中的 `checker/check` 针对额外的 speedrun、恢复训练和
窄模型对比合同，且使用与学习单不同的 1.65 损失门槛；本次没有声称通过该完整门槛。
实际采用其独立 NumPy evaluator 对全部尺度模型和 Part 2 权重重新评分，并单独运行
checker 自身的单元测试。
