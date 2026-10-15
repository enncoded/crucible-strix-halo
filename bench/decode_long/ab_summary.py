import re,glob,statistics as st,sys,os
d=os.path.expanduser('~/crucible-work/wt/decode-long-m/results/ab')
def one(tag):
    L=open(f'{d}/{tag}.log').read()
    m=re.search(r'decode (\d+) tok in ([\d.]+) s \(([\d.]+) tok/s\)',L)
    if not m or 'rc=0' not in L: return None
    secs=float(m.group(2)); end=float(re.search(r'end=([\d.]+)',L).group(1))
    ids=re.search(r'^ids: (.*)$',L,re.M).group(1)
    S=[]
    for l in open(f'{d}/{tag}.clk'):
        p=l.split()
        if len(p)==3:
            try: S.append((float(p[0]),int(p[1])/1e6,int(p[2])/1e6))
            except: pass
    # decode window: the last `secs` seconds before the process end (exit time ~0.2s)
    w=[x for x in S if end-secs-0.3<=x[0]<=end-0.2]
    return dict(secs=secs,tps=float(m.group(3)),ids=ids,mhz=st.mean(x[1] for x in w) if w else float('nan'),
                w=st.mean(x[2] for x in w) if w else float('nan'),n=len(w))
for ctx in (131072,204800):
    print('ctx',ctx)
    rows=[]
    for i in range(1,5):
        a=one(f't12-{ctx}-{i}') if os.path.exists(f'{d}/t12-{ctx}-{i}.log') else None
        b=one(f'dl1-{ctx}-{i}') if os.path.exists(f'{d}/dl1-{ctx}-{i}.log') else None
        if a and b:
            print(f' pair {i}: t12 {a["secs"]:.3f}s {a["tps"]:.2f}tok/s {a["mhz"]:.0f}MHz {a["w"]:.1f}W | dl1 {b["secs"]:.3f}s {b["tps"]:.2f}tok/s {b["mhz"]:.0f}MHz {b["w"]:.1f}W | change {a["secs"]/b["secs"]*100-100:+.1f}% ids {"equal" if a["ids"]==b["ids"] else "DIFFER"}')
            rows.append((a,b))
    if rows:
        ch=[a['secs']/b['secs']*100-100 for a,b in rows]
        print(f' n={len(rows)} median change {st.median(ch):+.1f}% mean {st.mean(ch):+.1f}%; mean t12 {st.mean(a["tps"] for a,_ in rows):.2f} dl1 {st.mean(b["tps"] for _,b in rows):.2f} tok/s')
