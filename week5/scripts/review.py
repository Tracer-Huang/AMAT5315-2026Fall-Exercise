"""Build an offline, relative-link review page from measured evidence."""
from pathlib import Path
import json,html
root=Path(__file__).resolve().parents[1]
a=root/'artifacts'
r=json.loads((a/'reflector-checks.json').read_text());m=json.loads((a/'marmousi-checks.json').read_text())
sections=[
 ('1 · 自动微分','从实际计算图逐节点求导。黄色 add_any 节点汇总共享变量的两条贡献。','ad/modes.png','ad/graph.png','ad/grad-graph.png'),
 ('2 · 正向传播','震源发出脉冲，接收器记录压力。波峰位移检验与接收信号幅值均通过。','inputs.png','forward/gathers.png','forward/wave-speed.png','forward/wavefield.png'),
 ('3 · 伴随成像','转置恒等式检验正反算子是否一致；图像峰值正确落在 2.1 km。旁瓣不是额外地层。','adjoint/image.png','adjoint/wavefield.png'),
 ('4 · 检查点','预算越小，保存状态越少，正向重算越多。四种预算与完整轨迹的图像误差均为零。','checkpoint-actions.png','checkpoint-work.png'),
 ('5 · Marmousi','上部倾斜结构可辨认。整幅图保持同一幅度标尺，无深度增益；深部仍弱。','marmousi.png')]
blocks=[]
for i,(title,desc,*figures) in enumerate(sections,1):
 imgs=''.join(f'<figure><a href="artifacts/{p}"><img loading="lazy" src="artifacts/{p}" alt="{html.escape(title)} · {p}"></a><figcaption>{p}</figcaption></figure>' for p in figures)
 blocks.append(f'<section id="part{i}"><h2>{title}</h2><p>{desc}</p>{imgs}</section>')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Week 5 · 本地审核</title>
<style>:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f5f2;color:#172b35;font:16px/1.75 system-ui,-apple-system,sans-serif}main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:38px;line-height:1.2;margin:12px 0}h2{font-size:25px}a{color:#006b6c;text-underline-offset:4px}.tag{font-size:12px;letter-spacing:.15em;color:#557070}.status{background:#fff3cf;border-left:4px solid #bb8a27;padding:15px 20px;margin:24px 0}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.metric{background:white;padding:20px;border:1px solid #dce3df;border-radius:10px}.metric strong{display:block;font-size:25px;color:#006b6c}.metric span{font-size:13px;color:#567}nav{display:flex;gap:16px;flex-wrap:wrap;margin:26px 0}section{background:white;border:1px solid #dce3df;border-radius:12px;padding:24px;margin:28px 0}figure{margin:24px 0}img{width:100%;height:auto;display:block;border:1px solid #eef0ee}figcaption{font-size:12px;color:#667;text-align:center;overflow-wrap:anywhere}.links{display:flex;gap:18px;flex-wrap:wrap}footer{font-size:13px;color:#567}@media(max-width:650px){main{padding:25px 14px}h1{font-size:30px}.metrics{grid-template-columns:1fr}section{padding:14px}}
</style><main><div class="tag">AMAT5315 / WEEK 05 / 2026-09-30</div><h1>从自动微分到地下成像</h1><p>JAX 手写求导 → Rust 波传播 → Enzyme 伴随 → Treeverse → Marmousi</p>
<div class="status"><b>技术提交材料已就绪 · 严格复核缺口已全部关闭</b><br>技术验收已通过，用户已授权按 PDF 要求提交至练习仓库。课堂展示、本人阅读理解与云账户准备保留为本人事项。</div>
<div class="metrics"><div class="metric"><strong>2.1 km</strong><span>反射层成像峰值，与输入深度一致</span></div><div class="metric"><strong>1.99 × 10⁻¹⁶</strong><span>Born / adjoint 转置相对误差</span></div><div class="metric"><strong>6 个状态</strong><span>Marmousi 保存状态 20.8 MB；不等于总内存</span></div></div>
<nav><a href="#part1">自动微分</a><a href="#part2">波传播</a><a href="#part3">伴随成像</a><a href="#part4">检查点</a><a href="#part5">Marmousi</a></nav>
<div class="links"><a href="STUDY_NOTES.zh.md">中文讲解</a><a href="AUDIT.md">逐项验收与待办</a><a href="README.md">复现说明 / 全部证据</a><a href="artifacts/course-check.txt">课程原始检查器结果</a><a href="vendor/viewer.html">打开本地波场查看器</a></div>
'''+''.join(blocks)+'''<section><h2>你审核时需要确认什么</h2><p>能否解释共享梯度为什么相加、完整状态为什么包含两个波场、一次 Enzyme JVP/VJP 怎样串联到整条轨迹，以及重放为什么不重复累加图像。</p><p>图像是 JᵀJm，能够定位结构，但有限带宽和观测覆盖限制了幅度与分辨率。数值检查通过不替代本人理解或老师验收。</p><p>下次课为 10 月 14 日。云账户应在老师说明费用后由本人注册与验证支付；GPU 等 Week 6 代码准备好再租。</p></section><footer>所有图由本次 JAX/Rust 输出生成，波场 PNG 来自课程原始查看器。资料位于 /Users/joshua/Downloads/amat5315/week5/。本页无外部脚本，不上传数据。</footer></main></html>'''
(root/'review.html').write_text(page)
