"""Build the offline Week 8 review from the measured evidence, without network access."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
escape = html.escape
c = json.loads((ROOT / 'evidence/cpu-measurements.json').read_text())
g = json.loads((ROOT / 'evidence/grokking-measurements.json').read_text())
v = json.loads((ROOT / 'evidence/verification.json').read_text())
p1, p2, p3 = c['part1'], c['part2'], c['part3b']
r = p3['result']
ab = g['mechanism']['ablation']


def picture(name, caption):
    return (f'<figure><a href="evidence/{name}"><img loading="lazy" src="evidence/{name}" '
            f'alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>')


cards = [
    (f"{p1['uniform']['loss']:.3f} → {p1['bigram']['loss']:.3f}", '利用字符频率与前一字符，验证损失下降'),
    (f"{p2['independent_score']['loss']:.3f}", 'Transformer 的完整验证损失；课程目标 < 1.85'),
    (f"{r['actual']:.3f}", '宽度 192 的两种子均值；落入事前预测区间'),
    (f"{ab['remove_key']['test_acc']:.3f}", '删除五个关键频率后的测试准确率'),
]
card_html = ''.join(f'<article><strong>{escape(a)}</strong><span>{escape(b)}</span></article>' for a, b in cards)
table = ''.join(f"<tr><td>{x['seed']}</td><td>{x['memorization_step']}</td>"
                f"<td>{x['grokking_step']}</td><td>{x['delay_ratio']:.1f}</td></tr>"
                for x in g['groups']['main']['runs'])
extra = ''.join(f"<tr><td>{width}</td><td>{x['prediction']:.4f} "
                f"[{x['interval_95'][0]:.4f}, {x['interval_95'][1]:.4f}]</td>"
                f"<td>{x['actual']:.4f}</td><td>{'是' if x['within_interval'] else '否'}</td></tr>"
                for width, x in p3['extra_heldout'].items())
samples = ''.join(f"<details><summary>温度 {s['temperature']}：500 个新字符</summary>"
                  f"<pre>{escape(s['text'])}</pre></details>" for s in p2['samples'])

body = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Week 8 · 本地审核</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f9;color:#243247;font:16px/1.7 system-ui,-apple-system,sans-serif}}
main{{max-width:1100px;margin:auto;padding:40px 24px}}h1{{font-size:36px;line-height:1.25}}h2{{font-size:25px;margin-top:46px}}
p{{max-width:940px}}a{{color:#245e9a}}.status{{display:inline-block;padding:4px 12px;border:1px solid #b8cbdc;background:#e6f1f1;border-radius:6px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}article{{background:white;border:1px solid #dce3ec;padding:18px}}
article strong,article span{{display:block}}article strong{{font-size:24px}}article span{{font-size:14px;color:#586679}}
figure{{margin:24px 0;background:white;border:1px solid #dce3ec;padding:14px}}img{{display:block;width:100%;height:auto}}
figcaption{{color:#586679;font-size:14px;text-align:center;margin-top:8px}}nav{{display:flex;gap:10px;flex-wrap:wrap}}
nav a{{background:white;border:1px solid #dce3ec;padding:8px 12px;text-decoration:none}}table{{border-collapse:collapse;width:100%;background:white}}
td,th{{text-align:left;padding:10px;border-bottom:1px solid #e1e7ee}}.table{{overflow-x:auto}}pre{{white-space:pre-wrap;background:#eaf0f7;padding:16px;overflow:auto}}
details{{background:white;border:1px solid #dce3ec;padding:12px;margin:10px 0}}summary{{cursor:pointer}}.note{{padding:16px;background:#fff5df;border-left:4px solid #d6a238}}
@media(max-width:760px){{.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:29px}}main{{padding:24px 16px}}}}
@media(max-width:420px){{.cards{{grid-template-columns:1fr}}}}
</style><main>
<span class="status">Week 8 · 2026-10-27 旧版计算证据</span>
<h1>字符预测、尺度规律与顿悟</h1>
<p class="note">课程已更新为 24 页新版。当前页面仅对应旧版，<a href="revised-20261028/review.html">请查看新版审核页</a>。</p>
<p>Week 8 的四个核心部分已按课程要求运行。下图使用本次本机训练和验证数据；教师资源包中的参考结果不作为本次输出。</p>
<nav><a href="../../../week8/week8-learning-sheet.pdf">旧版学习单 PDF（本地）</a><a href="STUDY_NOTES.zh.md">中文学习笔记</a>
<a href="README.md">设置与复现</a><a href="STRICT_AUDIT.zh.md">逐项复核</a><a href="CLOUD_CHALLENGE.zh.md">云端挑战说明</a></nav>
<p>自动核验：{v['passed']}/{v['total']} 项通过；图中的区间失准如实保留。课程要求下周课堂展示。</p>
<div class="cards">{card_html}</div>
<h2>1 · 下一字符的概率</h2><p>自回归模型把序列概率拆成条件概率的乘积。交叉熵是实际字符的平均负对数概率；困惑度为其指数。三种基线只用训练集拟合，验证集最后按字符数加权。</p>
{picture('cpu-part1-probabilities.png','字符频率和 t 之后的条件概率')}
{picture('cpu-part1-baselines.png','四个预测器的验证损失与困惑度')}
<h2>2 · 因果 Transformer</h2><p>改变后半窗口的字符，前半窗口的 logits 最大变化为 0；第一层四个注意力头均不读取未来。验证分数由独立 NumPy 前向重新计算。</p>
{picture('cpu-part2-loss.png','训练与验证损失：曲线从原日志恢复，保留四位小数')}
{picture('cpu-part2-attention.png','第一层四个注意力头；对角线以上为零')}
{samples}
<h2>3 · 先预测，再训练</h2><p>八个宽度拟合纯幂律，宽度 8 排除。宽度 192 的预测为 {r['predicted']:.4f} [{r['interval_95'][0]:.4f}, {r['interval_95'][1]:.4f}]，随后训练的平均损失为 {r['actual']:.4f}。</p>
{picture('cpu-part3-scaling.png','固定预算下的拟合、95% 区间与 192 宽度事前预测')}
<p class="note">学习单已经公开教师参考结果，所以时间戳只能证明本次预测先于本次训练，并不构成完全不知参考结果的盲法。</p>
<h3>可选扩展：外推失准</h3><p>256 和 384 的预测在两组训练之前冻结。它们的实测均未落入区间；384 甚至比 256 更差。固定训练预算可能影响比较，但本次不足以确定唯一原因。</p>
<div class="table"><table><tr><th>宽度</th><th>事前预测 [95% 区间]</th><th>实测</th><th>命中</th></tr>{extra}</table></div>
{picture('cpu-part3-scaling-extra.png','扩展到宽度 256 和 384 的事前预测与实测')}
<h2>4 · 模加法中的顿悟</h2><p>训练对很快被记住，未见过的输入对却在许多步之后才被正确预测。参数范数、无衰减、更多训练对和随机标签共同说明训练与泛化可以不同步。</p>
<div class="table"><table><tr><th>种子</th><th>记忆步数</th><th>顿悟步数</th><th>延迟比</th></tr>{table}</table></div>
{picture('grokking-curves.png','五个主种子的准确率、损失和参数范数')}
{picture('grokking-controls.png','权重衰减、训练比例和随机标签对照')}
<p>seed 0 只保留五个关键频率时准确率 {ab['keep_key']['test_acc']:.5f}；删除它们后为 {ab['remove_key']['test_acc']:.5f}。独立 NumPy 实现复算了频谱和消融。</p>
{picture('grokking-fourier.png','初始化、记忆和训练末态的残数嵌入傅里叶功率')}
{picture('grokking-ablation.png','保留或去除关键频率与随机非关键频率')}
<p><a href="evidence/verification.json">机器可读核验记录</a> · <a href="evidence/grokking-independent-check.txt">独立 NumPy 复算</a> · <a href="README.md">全部复现命令</a></p>
</main></html>'''
(ROOT / 'review.html').write_text(body, encoding='utf-8')
print('wrote', ROOT / 'review.html')
