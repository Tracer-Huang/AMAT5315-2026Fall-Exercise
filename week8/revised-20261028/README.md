# Week 8 新版：语言模型与 Transformer

本目录对应 **2026-10-28 16:56:53** 的 24 页学习单，SHA-256
`4e33bea929459c01511efc0298308a93d08d99ee7df38372bbf52e89be69cc35`。
它替换了原先的 12 页学习单；旧版练习和 Git 提交保留在上一级目录，不是本版的完成证据。
从 [本地审核页](review.html)、[中文学习笔记](STUDY_NOTES.zh.md)或
[逐项复核](AUDIT.zh.md)开始。

课程要求在 Week 9 开始时展示，注明没有另行上传的作业；PDF 同时要求推送源码和证据。
两部指定视频需学习者本人观看。课程页面在本次检查时未提供新版资源 ZIP。

## 六部分结果

1. **嵌入。** potion-base-8M 中三组关系的留一余弦均值为 0.4659、0.5485、0.4330；
   seed 1730 打乱行后的对照为 0.0295、−0.0148、0.0088。三个真实值均 ≥0.3、对照
   均在 ±0.15 内。
2. **n-gram。** 语料切分为 24,867 / 2,763 个训练/验证故事；验证集仅 3 个故事的
   tokenizer 解码与原文不同。均匀、1–6 gram 的 bits/字符分别为 4.386、3.403、
   2.136、1.473、1.310、1.464、1.793；4-gram 最优。显式枚举长度 1–3 的 bigram
   总概率与 1 的差小于 10⁻¹²；250,000 token 的采样检验 p=0.2543。
3. **权重中的记忆。** d=64 的外积与最小二乘容量分别约 66.9、128.5；五个容量曲线
   的 d 方向拟合斜率依次为 0.988、1.003、1.042、1.906、2.767。1-token MLP 的
   验证损失 3.04371，比 bigram 的 3.03819 高 0.00552 nats/token，小于 0.01。
4. **注意力的文本内记忆。** [`part4-picture-recall.png`](evidence/part4-picture-recall.png)
   由本次随机键和值实际生成，16 个数值及其来源在 `part4-picture.json`。M=4096 时
   外积、ReLU、ReLU²、softmax 的 R 分别为 8.137、5.713、1.209、0.0565。
5. **Transformer。** 两层、宽 96、四头、上下文 64，训练 6,000 步。完整验证得到
   1.74992 nats/token、**1.23042 bits/字符**，低于 4-gram 的 1.30975。五种采样
   分布的卡方检验 p 均 >0.001；自采样惊讶度与熵的 z 在 T=1 为 0.186，在 T=0.7 为
   −78.026。完整文本样例及统计见 `part5-transformer.json`。
6. **Chinchilla 拟合。** 245 条记录剔除最高损失 5 条；27 个初始点拟合并完成 400 次
   bootstrap。α=0.34731±0.01512、β=0.36717±0.02136、E=1.81722±0.02499；
   预算 5.76×10²³ FLOP 时 D*/N*=17.92，95% bootstrap 区间 [7.08, 36.95]。

Part 3 的容量实验每个维度和 M 使用 3 个独立随机样本；当 M≤256 时查询所有存储键，
当 M>256 时查询每次生成的前 256 个存储键。图中是先对各次查询取中位数、再对三次
独立抽样取中位数的结果；不是声称逐一查询了大 M 下的全部键。Part 4 图则每行使用
一次独立样本，点面积在每个面板内与归一化回忆值的正部成正比。

两个 PDF Extension 也已计算：八个注意力头的权重图，以及 MLP 每个隐藏单元两侧的
前五输入/输出 token。MLP 侧的前五激活占比中位数本次为 **训练后 0.209、初始化
0.137**；这与 PDF 示例中“训练后低于初始化”的观察方向不同。我们保留自己的结果，
不据此声称复现了示例的该定性结论。

## 图与对应命令

每条命令在本目录执行；下表的绘图脚本读取本次已保存的运行，训练脚本只在全新运行
目录中使用，避免覆盖权重和原始结果。

| 证据图 | 生成命令 |
| --- | --- |
| [词向量关系](evidence/part1-relations.png) | `uv run python embeddings.py` |
| [n-gram 损失与未见比例](evidence/part2-ngram-ladder.png) | `uv run python ngrams.py` |
| [干扰比](evidence/part3-interference.png)、[容量](evidence/part3-capacity.png) | `uv run python capacity.py` |
| [MLP 损失](evidence/part3-mlp-loss.png) | `uv run python plot_mlp.py`（先运行 `mlp.py`） |
| [MLP 隐藏单元 Extension](evidence/part3-mlp-units.png) | `uv run python interpret_mlp.py` |
| [四种记忆的回忆图](evidence/part4-picture-recall.png) | `uv run python part4_picture.py` |
| [模型比较](evidence/part5-model-comparison.png)、[位置损失](evidence/part5-context-loss.png) | `uv run python evaluate_transformer.py --version 1` |
| [八个注意力头 Extension](evidence/part5-attention-heads.png) | `uv run python attention_maps.py` |
| [Chinchilla 拟合](evidence/part6-chinchilla-fit.png)、[残差](evidence/part6-residuals.png)、[最优分配](evidence/part6-optimal.png) | `uv run python chinchilla.py` |

实际本机 Transformer 评估为 `--version 3`：第一次选的上下文过于集中，两个截断分布
只有一个类别而无法做卡方检验；第二版选定宽分布后统计有效；第三版修正 trigram
生成最初上下文不足时使用可用的较短 n-gram。旧版评估保存在 `artifacts/transformer/`
下，训练权重未重跑。新版上下文在采样前按 64 个固定验证窗口的预测熵选取，未按
卡方结果挑选。

## 数据、环境和复现

Python 3.14、CPU JAX 0.10.2、Optax 0.2.8、NumPy 2.4.6、SciPy 1.18.1；完整依赖锁见
`pyproject.toml` 与 `uv.lock`。源数据与权重、缓存、原始运行均由 `.gitignore` 忽略。
TinyStories 的下载地址是：

`https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt`

其他固定输入为 `minishlab/potion-base-8M` revision `bf8b056` 的 `model.safetensors` 和
`tokenizer.json`；`karpathy/tinyllamas` revision `0bd21da` 的 `stories260K/tok512.model`；
以及 [Chinchilla CSV](https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv)。
原始文件的 SHA-256、故事与 token 统计见 `artifacts/data-manifest.json`。语料、模型和
数据不应被提交到 Git。

在复制了源码及上述五个输入文件的**新目录**执行：

```sh
uv sync --locked
mkdir -p data artifacts evidence
uv run python pipeline.py
uv run python embeddings.py
uv run python ngrams.py
uv run python bigram_sample_audit.py
uv run python capacity.py
uv run python mlp.py
uv run python plot_mlp.py
uv run python interpret_mlp.py
uv run python part4_picture.py
uv run python train_transformer.py
uv run python evaluate_transformer.py --version 1
uv run python attention_maps.py
uv run python chinchilla.py
uv run python -m pytest tests -q
uv run python audit_revised.py
```

输入文件依次命名为 `data/TinyStoriesV2-GPT4-valid.txt`、`data/tok512.model`、
`data/potion-model.safetensors`、`data/potion-tokenizer.json`、`data/chinchilla-runs.csv`。
`pipeline.py`、训练和图片运行在输出已存在时拒绝覆盖；请用新目录复现。MLP 将每批
16,384 个真实采样对按 `(输入,目标)` 汇总后计算损失，测试验证了这一做法与逐例损失
及梯度一致，优化器仍使用该批次的所有样本。

验证结果见 [audit.json](evidence/audit.json) 和 [tests.txt](evidence/tests.txt)。独立
NumPy 前向核对两个实际验证窗口，不声称已独立重算 Transformer 的全部验证集。
