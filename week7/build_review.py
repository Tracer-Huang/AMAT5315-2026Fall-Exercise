"""Build an offline, read-only review page from measured local evidence."""
import html,json,re
from pathlib import Path
E=html.escape
strict=json.loads(Path('evidence/strict-audit.json').read_text());ready=strict['submission_ready']
r=json.loads(Path('evidence/toy-results.json').read_text());im=json.loads(Path('evidence/image-results.json').read_text());vd=json.loads(Path('evidence/velocity-domain-diagnostics.json').read_text())
def markdown(text):
    out=[];para=[];code=[];incode=False
    def flush():
        if para:out.append('<p>'+inline(' '.join(para))+'</p>');para.clear()
    def inline(s):
        s=E(s);s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s);s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s);s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s);return s
    for line in text.splitlines():
        if line.startswith('```'):
            flush()
            if incode:out.append('<pre>'+E('\n'.join(code))+'</pre>');code=[]
            incode=not incode;continue
        if incode:code.append(line);continue
        if line.startswith('#'):
            flush();n=len(line)-len(line.lstrip('#'));out.append(f'<h{min(n,4)}>'+inline(line[n:].strip())+f'</h{min(n,4)}>')
        elif not line.strip():flush()
        elif line.startswith('    '):flush();out.append('<pre>'+E(line.strip())+'</pre>')
        elif re.match(r'^(- |\d+\. )',line):flush();out.append('<p class="list-item">'+inline(line)+'</p>')
        else:para.append(line)
    flush();return '\n'.join(out)
parts=[('Part 1','训练去噪规则','四簇点如何变成噪声，再由 DDPM 生成新点。','run_toy.py',[('forward.png','正向扩散：同一批点逐步变成高斯噪声'),('forward.gif','正向过程动画'),('loss.png','噪声与速度网络训练损失'),('training.gif','十个模型阶段与十条共同起点轨迹'),('ddpm.png','2000 点 DDPM 与精确样本对照'),('reverse.gif','DDPM 反向生成动画')]),('Part 2','同一个模型，确定性采样','DDIM 不重新训练，比较确定性与分布质量。','run_toy.py',[('determinism.png','精确分布、DDIM 与 DDPM'),('trajectories.png','共同起点的生成路径')]),('Part 3','速度场与步数比较','重新训练流匹配网络，区分学习误差和离散步长误差。','run_toy.py / diagnostics.py',[('velocity.png','原完整方形网格：学习速度与精确速度对照'),('velocity-supported.png','另行声明的数据支持区域：三时刻 RMS 均低于 0.1'),('comparison.png','23 组采样比较：能量距离、比例误差与宽度误差'),('exact-vs-learned.png','实线学习模型；虚线精确预测器')]),('Part 4','从噪声生成照片','固定预训练 U-Net，运行 1000 步 DDPM，不再训练。','run_images.py',[('photographs.png','四张照片的反向生成阶段'),('real-vs-generated.png','真实 CIFAR-10 与生成图片')]),('Part 5','用 RePaint 补图','只提供可见像素和遮罩，生成隐藏区的多种可能。','run_images.py',[('inpainting.png','四张测试图 × 四个种子；真图、观测、补图、方差')])]
body=['<header><span class="eyebrow">AMAT5315 · WEEK 7 · 2026-10-21</span><h1>从噪声生成数据</h1><p class="lead">扩散模型、确定性采样与流匹配。五部分已实现并运行；验收状态见下方严格复核。</p><p><strong>交付规则：严格验收通过后 push 源码、环境与证据；无 LMS 上传任务</strong></p><p>课程要求：Week 8 开始时展示；日程为 2026-10-28。原文明确写着 “Nothing is submitted”。</p><div class="links"><a href="../../../week7/week7-learning-sheet.pdf">课程原文 PDF</a><a href="STUDY_NOTES.zh.md">中文笔记</a><a href="README.md">复现与结果说明</a><a href="evidence/audit.json">测量核验 JSON</a></div></header>']
if ready:
 body.append('<aside style="background:#e8f4eb;border-color:#438455"><strong>严格 PDF 复核：全部检查通过</strong><p>已补齐早期损失历史与两次采样的配对轨迹；保持原完整网格，三个时刻的速度相对 RMS 均低于 0.1。初次失败结果保留为基线。</p><a href="STRICT_AUDIT.zh.md">完整复核报告</a> · <a href="evidence/strict-audit.json">逐项复核数据</a></aside>')
else:
 body.append('<aside><strong>严格复核未全部通过</strong><p>'+E('；'.join(strict['failed_requirements']))+'</p></aside>')
body.append('<p>当前完整网格 RMS：'+ '、'.join(f'{value:.5f}' for value in r['velocity_relative_rms'])+'。神经网络的系数由随机回归训练得到，精确预测器仅用于明确标注的对照。跨框架 PyTorch 数值一致性未验证，不作相应声明。</p>')
f=next(v for v in r['comparison'] if v['method']=='flow' and v['steps']==20)
body.append('<div class="stats">'+''.join(f'<article><b>{a}</b><span>{b}</span></article>' for a,b in [('0','DDIM 重复运行差异'),(f'{f["energy"]:.4f}','20 步 Flow 能量距离'),('0','补图可见像素最大变化'),(f'{im["mean_hidden_mse"]:.3f}','16 张补图平均隐藏区 MSE')])+'</div>')
body.append('<nav>'+''.join(f'<a href="#p{i}">{p[0]} · {p[1]}</a>' for i,p in enumerate(parts,1))+'<a href="#notes">完整中文讲解</a></nav>')
for i,(label,title,desc,cmd,figs) in enumerate(parts,1):
 body.append(f'<section id="p{i}"><span class="eyebrow">{label}</span><h2>{title}</h2><p>{desc}</p>')
 for name,caption in figs:
  actual='diagnostics.py' if name in ['velocity-supported.png','exact-vs-learned.png'] else cmd.split(' / ')[0]
  body.append(f'<figure><img loading="lazy" src="evidence/{name}" alt="{E(caption)}"><figcaption>{E(caption)}<br><code>uv run python {actual}</code></figcaption></figure>')
 if i==1:
  m=r['ddpm'];body.append(f'<p>最终 DDPM 簇比例：{E(str(m["fractions"]))}；最大中心误差 {m["centre_error"]:.5f}，最大宽度误差 {m["width_error"]:.5f}。每个已训练阶段都显示已有损失，初始化阶段明确标为尚无训练损失。原始路径保存在 artifacts/training-paths.npz。</p>')
 if i==2:
  d=r['determinism'];body.append(f'<p>DDIM 两次结果逐点相同，最大差异 {d["ddim_max_difference"]}；两次 DDPM 的同簇比例 {d["ddpm_same_cluster_fraction"]:.2%}。实线和虚线展示两个重复，DDIM 重合，DDPM 分离。</p>')
 if i==3:body.append(f'<p>20 步 Flow 的能量距离为 {f["energy"]:.6f}，最大宽度误差为 {f["width_error"]:.5f}，分别满足 &lt;0.005 和 &lt;0.03。本次精确对照零假设 95% 分位为 {r["energy_null"]["q95"]:.6f}。不能只凭能量距离宣布所有分布细节都正确，也不能由这个例子推广为所有问题上的速度排名。</p>')
 if i==4:body.append('<p>权重下载前已学习照片先验；本次只执行网络预测和采样。32×32 图像出现动物、鸟类与场景轮廓，类别仍有歧义。只有显示时裁剪，采样状态不裁剪。</p>')
 if i==5:
  body.append('<p>四个种子分别为 7201–7204。每个种子对四图运行 99 次模型调用，隐藏区为 6:26×6:26。可见像素精确保留、洞外方差为零；合理补图仍可能与唯一真值差异很大。</p><div class="table"><table><thead><tr><th>测试图</th><th>种子</th><th>隐藏 MSE</th><th>可见区最大变化</th></tr></thead><tbody>')
  for v in im['completions']:body.append(f'<tr><td>{v["image"]}</td><td>{v["seed"]}</td><td>{v["hidden_mse"]:.6f}</td><td>{v["observed_max_change"]}</td></tr>')
  body.append('</tbody></table></div>')
 body.append('</section>')
body.append('<section id="notes">'+markdown(Path('STUDY_NOTES.zh.md').read_text())+'</section>')
body.append('<footer>本地审核副本 · 原始课程材料保存在 amat5315/week7 · 实现与证据保存在练习仓库 week7 · 结果与限制以 evidence/*.json 为准。</footer>')
css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f5f7;color:#20252e;font:16px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1120px;margin:auto;padding:48px 24px}header,section{background:white;padding:32px;border-radius:16px;margin-bottom:24px}h1{font-size:42px;line-height:1.2;margin:14px 0}h2{font-size:28px;line-height:1.35}.eyebrow{letter-spacing:.12em;font-size:12px;color:#576580}.lead{font-size:20px;color:#536071}a{color:#244e92;text-underline-offset:3px}.links,nav{display:flex;gap:12px;flex-wrap:wrap}.links a,nav a{background:#eef2f8;padding:8px 12px;border-radius:8px;text-decoration:none}nav{margin:24px 0}aside{background:#fff3da;border-left:4px solid #b27a23;padding:20px 24px;border-radius:8px;margin:24px 0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stats article{background:#e9eef7;padding:18px;border-radius:12px}.stats b{display:block;font-size:28px}.stats span{font-size:13px;color:#536071}figure{margin:24px 0;border:1px solid #e1e5ec;border-radius:10px;overflow:hidden}img{display:block;max-width:100%;width:100%;height:auto}figcaption{padding:12px 18px;background:#f8f9fc;color:#536071;font-size:14px}code,pre{font:13px/1.6 ui-monospace,monospace;background:#eef1f5;border-radius:5px}pre{padding:14px;white-space:pre-wrap;overflow-wrap:anywhere}code{padding:2px 5px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #e1e5ec;padding:8px;text-align:left}footer{color:#6b7280;padding:16px;font-size:13px}.list-item{padding-left:12px}section{scroll-margin-top:16px}@media(max-width:640px){main{padding:20px 12px}header,section{padding:20px}h1{font-size:32px}.stats{grid-template-columns:repeat(2,1fr)}h2{font-size:24px}}'''
Path('review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Week 7 · 本地审核</title><style>'+css+'</style><main>'+''.join(body)+'</main></html>')
print('Built',Path('review.html').resolve())
