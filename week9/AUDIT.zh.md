# Week 9：按学习单复核

核对依据：课程 `week9-learning-sheet.pdf` 第 6–15 页。运行环境为 Rust 1.98.1、CPU f64；用户允许使用高于 1.96.0 的版本。

| 要求 | 对照证据 |
|---|---|
| 八表 ASIA、枚举与收缩、三组证据及 T–L 联合分布 | `artifacts/{prior,AD,ADX}/run.json`；证据质量 1、0.004501375、0.00098822675；最大边缘误差 < 10⁻¹⁰、相对质量误差 < 10⁻¹⁰。 |
| T,A 轴序检查 | `asia.uai` scope 为 `2 1 0`；T=0 时 A=0、1 的值分别为 0.05、0.01；解析使用最后一轴变化最快。 |
| 查询图、概率叠加、联合分布与乘积的差值 | `evidence/graphs.svg`、`posteriors.png`、`joint.png`；图从实际输出生成。 |
| 矩阵、迹与三自旋 Ising 比较 | `check_numerics.py` 打印 tenferro 与 NumPy/枚举结果及绝对误差；`evidence/numerical-checks.json`。误差均 < 10⁻¹²。 |
| 随机/贪心计划、保存、独立成本、重放与无需收缩的分析 | PDF 命令对应 `runs/{random,greedy,replay,analysis}`；成本逐步记录，树、种子、模型和观测绑定在 `order.json`。随机 96/16，贪心 54/8，重放质量相对误差 < 10⁻¹⁰。 |
| BP 设置、每步信念、无阻尼残差及累计工作 | `artifacts/bp-{prior,AD,ADX}`；非均匀树测试通过；ADX 第 93 步残差 8.225×10⁻¹¹，最大误差 0.01061。 |
| Promedus 贪心和 TreeSA、PR/MAR、成本及 BP 对照 | `artifacts/promedus-{greedy,treesa,bp}` 与 `verification.json`。两计划 log10 Z 差 < 10⁻¹⁰；PR 误差 < 10⁻⁴，MAR 误差 < 10⁻⁵。BP 在 1,000 步仍未收敛且指定查询误差约 0.38235。 |
| README、来源及重现 | `README.md`、`fetch_data.sh` 和锁定依赖 `Cargo.lock`。原始 Promedus 数据保留在本地忽略目录，脚本按固定版本取回。 |

执行了 `cargo fmt --all --check`、`cargo test --locked`、Python 独立测试、`verify.py` 和课程要求的四条计划命令。安装位置为项目内忽略的 `.install/`，未替换全局程序。`benchmark` 仍用于 Promedus 多查询；PDF 指定的 ASIA 计划接口由 `infer` 实现。BP 的完整大型轨迹以 gzip 形式随证据保存。
