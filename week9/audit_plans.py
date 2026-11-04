import json,math
from pathlib import Path

def inspect(t, dims, steps):
    if 'Leaf' in t: return
    node=t['Node']
    for child in node['args']: inspect(child,dims,steps)
    e=node['eins']; labels=set().union(*map(set,e['ixs']))
    steps.append({'inputs':e['ixs'],'retained':e['iy'],'work':math.prod(dims[x] for x in labels),'entries':math.prod(dims[x] for x in e['iy'])})

def audit(folder,dims):
    result=json.loads((folder/'result.json').read_text()); outputs=[]
    for r in result['queries']:
        q=r['query']; p=folder/('plan-z.json' if not q else f'plan-{q[0]}.json')
        steps=[];inspect(json.loads(p.read_text()),dims,steps)
        work=sum(x['work'] for x in steps); largest=max(x['entries'] for x in steps)
        outputs.append({'query':q,'work':work,'largest_generated':largest,'steps':steps,'reported_work':2**r['work_log2'],'reported_space':2**r['space_log2']})
    (folder/'independent-costs.json').write_text(json.dumps(outputs,indent=2))
    return outputs
if __name__=='__main__':
    for f,n in [('omeco-AD',8),('omeco-random',8),('promedus-treesa',414),('promedus-greedy',414)]:
        p=Path('artifacts')/f; a=audit(p,[2]*n)
        print(f,[(x['query'],x['work'],x['largest_generated'],round(x['reported_work'])) for x in a])
