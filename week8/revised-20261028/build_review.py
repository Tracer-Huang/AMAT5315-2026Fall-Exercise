"""Build an offline Chinese review page from the actual saved measurements."""
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
E=html.escape
def read(name):return json.loads((ROOT/'evidence'/name).read_text())
p1=read('part1-relations.json');p2=read('part2-ngrams.json');cap=read('part3-capacity.json')
mlp=read('part3-mlp.json');units=read('part3-mlp-units.json');pic=read('part4-picture.json')
p5=read('part5-transformer.json');p6=read('part6-chinchilla.json');audit=read('audit.json')


def figure(file,caption):
    return f'<figure><a href="evidence/{file}"><img loading="lazy" src="evidence/{file}" alt="{E(caption)}"></a><figcaption>{E(caption)}</figcaption></figure>'


def printable(s):
    return ''.join(ch if ch in '\n\t' or ord(ch)>=32 else f'\\x{ord(ch):02x}' for ch in s)


samples=''.join(f"<details><summary>{E(name)} · seed {item['seed']} · 60 token</summary><pre>{E(printable(item['text']))}</pre></details>"
                for name,item in p5['comparison_samples'].items())
generations=''.join(f"<details><summary>{E(name)} · seed {item['seed']}</summary><pre>{E(printable(item['text']))}</pre></details>"
                    for name,item in p5['generated_samples'].items())
parts=[
 ('1','词向量里的关系','训练好的表把 token 映射为向量。留一关系方向在真表中一致，打乱表后趋近零。',
  [('part1-relations.png','三组词对：真表与打乱行对照')]),
 ('2','计数模型与上下文','训练故事内选平滑参数，验证故事独立评分。更长上下文降低损失，但稀疏计数最终使 5/6-gram 变差。',
  [('part2-ngram-ladder.png','1–6 gram 验证损失与未见 n-gram 比例')]),
 ('3','权重里的记忆',f"五种存储的容量随维度增长。1-token MLP 验证损失 {mlp['heldout_loss']:.4f}，距 bigram {mlp['difference']:.4f} nats/token。",
  [('part3-interference.png','d=64 的干扰比及理论线'),('part3-capacity.png','容量与维度：三次独立样本的中位数'),
   ('part3-mlp-loss.png','MLP 训练与完整验证比较'),('part3-mlp-units.png','Extension：四个高集中度隐藏单元')]),
 ('4','从当前文本构建记忆','四种存储分别回忆 32×32“算”字。每行是独立随机样本；红色表示干扰比 R<1，灰色表示 R≥1。',
  [('part4-picture-recall.png','外积、ReLU、ReLU² 与 softmax 的 4×4 回忆比较')]),
 ('5','Transformer 训练与生成',f"两层、宽 96 的因果模型达到 {p5['heldout']['bpc']:.3f} bits/字符，优于 4-gram 的 {p2['models']['4']['bpc']:.3f}。",
  [('part5-model-comparison.png','相同验证文本上的 bits/字符'),('part5-context-loss.png','位置损失与上下文长度'),
   ('part5-attention-heads.png','Extension：两层共八个注意力头')]),
 ('6','算力如何分配',f"240 条训练记录的稳健拟合 α={p6['parameters']['alpha']:.3f}、β={p6['parameters']['beta']:.3f}；指定预算 D*/N*={p6['tokens_per_parameter']:.1f}。",
  [('part6-chinchilla-fit.png','拟合曲线与实测比较'),('part6-residuals.png','拟合与论文打印参数的残差'),
   ('part6-optimal.png','模型大小和训练 token 的计算最优分配')]),
]
body=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Week 8 新版 · 本地审核</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#233247;font:16px/1.7 system-ui,-apple-system,sans-serif}
main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:36px;line-height:1.25}h2{font-size:25px;margin-top:45px}
p{max-width:950px}a{color:#245e9a}.status{display:inline-block;padding:4px 12px;border:1px solid #b4cbd3;background:#e5f2f2;border-radius:6px}
nav{display:flex;gap:10px;flex-wrap:wrap}nav a{background:white;border:1px solid #dce3ec;padding:8px 12px;text-decoration:none}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}article{background:white;border:1px solid #dce3ec;padding:18px}
article strong,article span{display:block}article strong{font-size:24px}article span{font-size:14px;color:#586679}
figure{margin:24px 0;background:white;border:1px solid #dce3ec;padding:14px}img{display:block;width:100%;height:auto}
figcaption{color:#586679;font-size:14px;text-align:center;margin-top:8px}.note{padding:16px;background:#fff3df;border-left:4px solid #d3a04e}
details{background:white;border:1px solid #dce3ec;padding:12px;margin:10px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow:auto}
@media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}main{padding:24px 16px}}
@media(max-width:420px){.cards{grid-template-columns:1fr}}</style><main>
<span class="status">2026-10-28 新版 · 课程任务与计算证据</span><h1>语言模型与 Transformer</h1>
<p>课程 PDF 已从旧版 12 页更新为新版 24 页。以下图表来自按新版要求进行的本机计算；旧版提交保留，不能作为新版作业的证明。</p>
<nav><a href="https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week8-learning-sheet.pdf">课程 PDF（在线）</a><a href="README.md">结果与复现</a><a href="STUDY_NOTES.zh.md">中文学习笔记</a><a href="AUDIT.zh.md">逐项复核</a><a href="../review.html">旧版审核页</a></nav>''']
body.append(f"<p>数值检查：{audit['passed']}/{audit['total']} 项通过。课程视频和 Week 9 的课堂展示仍需本人完成。</p>")
cards=[('0.466 / 0.549 / 0.433','三组词关系的平均余弦'),('1.230','Transformer 验证 bits/字符'),
       (f"{pic['ratios'][3][3]:.4f}",'4,096 条记忆时 softmax 的 R'),(f"{p6['tokens_per_parameter']:.1f}",'指定预算下最优 token / 参数')]
body.append('<div class="cards">'+''.join(f'<article><strong>{E(a)}</strong><span>{E(b)}</span></article>' for a,b in cards)+'</div>')
body.append('<p class="note">容量曲线在 M&gt;256 时每次查询固定的 256 个存储键；MLP 单元扩展的集中度方向与课程参考示例不同。完整边界见逐项复核。</p>')
body.append('<h2>你指出的回忆图</h2>'+figure('part4-picture-recall.png','M=64、256、1024、4096；四列分别为外积、ReLU、ReLU²、softmax。'))
for number,title,description,figures in parts:
    body.append(f'<section><h2>{number} · {E(title)}</h2><p>{E(description)}</p>')
    for file,caption in figures:
        if file!='part4-picture-recall.png':body.append(figure(file,caption))
    if number=='3':
        body.append(f"<p>Extension：训练后前五激活占比的中位数 {units['trained_median_share']:.3f}，初始化 {units['initial_median_share']:.3f}；与 PDF 示例的方向不同，保留实测值。</p>")
    if number=='5':
        body.append('<h3>五种模型的 60-token 样例</h3>'+samples+'<h3>Transformer 的四种采样设置</h3>'+generations)
        body.append('<p>卡方检验的上下文在抽样前按预测熵选择；第一版分布过于集中，失败记录保留在本机 artifacts。</p>')
    body.append('</section>')
body.append('<p><a href="evidence/audit.json">机器可读核验</a> · <a href="README.md">全部产生命令与数据版本</a></p></main></html>')
(ROOT/'review.html').write_text('\n'.join(body),encoding='utf-8')
print('wrote',ROOT/'review.html')
