"""Self-contained HTML report for a run (no network, no external assets)."""

from __future__ import annotations

import csv
import gzip
import html
import json
import math
from pathlib import Path

import numpy as np


def _load(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    verdict = json.loads((run_dir / "finalists.json").read_text())
    nets = np.load(run_dir / "nets.npy")
    hold_path = run_dir / "holdout.json"
    holdout = json.loads(hold_path.read_text()) if hold_path.exists() else {"looks": []}
    with gzip.open(run_dir / "ledger.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    return {"config": config, "verdict": verdict, "nets": nets, "holdout": holdout, "ledger": rows}


def _finite(x: float) -> float | None:
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def build(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    d = _load(run_dir)
    cfg, v, nets, holdout = d["config"], d["verdict"], d["nets"], d["holdout"]
    ppy = cfg["periods_per_year"]
    ann = math.sqrt(ppy)
    T, hold_start = cfg["index"]["T"], cfg["index"]["hold_start"]
    first = np.datetime64(cfg["dates"]["train"][0])
    dates = [str(first + i) for i in range(T)]

    # Trial Sharpe distribution (annualised) and search progress.
    sh = np.array([float(r["train_sharpe"]) for r in d["ledger"]])
    gen = np.array([int(r["generation"]) for r in d["ledger"]])
    ok = np.isfinite(sh)
    hist_counts, edges = np.histogram(sh[ok] * ann, bins=40) if ok.any() else (np.array([]), np.array([0.0]))
    progress = []
    best = -np.inf
    for g in range(int(gen.max()) + 1 if len(gen) else 0):
        m = ok & (gen == g)
        if m.any():
            best = max(best, float(sh[m].max()))
        progress.append([g + 1, best * ann if math.isfinite(best) else None])

    # Equity curves: combined survivors (with holdout once unsealed) and the top finalists.
    curves = []
    survivors = v["survivors"]
    if survivors:
        w = np.array(v["weights"])
        combo = np.nansum(nets[survivors] * w[:, None], axis=0)
        last = next((lk for lk in reversed(holdout["looks"]) if lk.get("combined_net")), None)
        full = np.full(T, np.nan)
        full[: len(combo)] = combo
        if last:
            hs = hold_start
            full[hs: hs + len(last["combined_net"])] = last["combined_net"]
        curves.append({"name": f"Survivors combined ({len(survivors)})", "values": full})
    for k in range(min(3 - len(curves), len(nets))):
        full = np.full(T, np.nan)
        full[: nets.shape[1]] = nets[k]
        curves.append({"name": f"Finalist #{k + 1}: {v['finalists'][k]['formula']}", "values": full})
    for c in curves:
        vals = c["values"]
        started = np.isfinite(vals)
        cum = np.cumsum(np.nan_to_num(vals))
        c["values"] = [round(float(x), 6) if s or i > 0 and started[:i].any() else None
                       for i, (x, s) in enumerate(zip(cum, started))]
        # stop the line where data ends (end of pre-holdout, unless unsealed)
        lastidx = int(np.nonzero(started)[0].max()) if started.any() else -1
        c["values"] = [x if i <= lastidx else None for i, x in enumerate(c["values"])]

    payload = {
        "dates": dates,
        "split": {"val": cfg["index"]["val_start"], "hold": hold_start},
        "curves": [{"name": c["name"], "values": c["values"]} for c in curves],
        "hist": {"counts": hist_counts.tolist(), "edges": edges.tolist(),
                 "luck": _finite(v["sr0_train"] * ann)},
        "pbo": v["pbo"]["logits"] if v["pbo"] else [],
        "progress": progress,
    }

    looks = holdout["looks"]
    tiles = [
        (f"{v['n_trials']:,}", "formulas tried", "every distinct formula counts against the winners"),
        (str(v["n_finalists"]), "finalists", "diverse best of the search (|corr| ≤ "
         f"{cfg['search']['max_corr']})"),
        (str(len(survivors)), "survivors", "passed every test below"),
    ]
    if v["pbo"]:
        tiles.append((f"{v['pbo']['pbo']:.2f}", "PBO", "probability the search overfit (0.5 = coin flip)"))
    if v["snoop"]:
        tiles.append((f"{v['snoop']['spa_p']:.3f}", "SPA p-value", "is the best finalist better than luck?"))
    if looks and looks[-1].get("combined"):
        c = looks[-1]["combined"]
        tiles.append((f"{c['sharpe_annual']:+.2f}", "holdout Sharpe /yr", f"combined survivors, {c['days']} days"))

    def row(i: int, f: dict) -> str:
        tr, va = f["train"], f["validation"]
        status = ('<span class="st good">✓ pass</span>' if f["passed"]
                  else f'<span class="st bad" title="{html.escape("; ".join(f["reasons"]))}">✕ fail</span>')
        cell = lambda x, fmt: "–" if x is None or not math.isfinite(x) else format(x, fmt)  # noqa: E731
        return (f"<tr><td class=num>{i + 1}</td><td class=formula><code>{html.escape(f['formula'])}</code></td>"
                f"<td class=num>{cell(tr['sharpe'] * ann, '+.2f')}</td><td class=num>{cell(tr['dsr'], '.2f')}</td>"
                f"<td class=num>{cell(va['sharpe'] * ann, '+.2f')}</td><td class=num>{cell(va['dsr'], '.2f')}</td>"
                f"<td>{status}</td><td class=why>{html.escape('; '.join(f['reasons']))}</td></tr>")

    look_rows = "".join(
        f"<li>{html.escape(lk['time'])}: {len(lk['survivors'])} survivors"
        + (f", combined holdout Sharpe {lk['combined']['sharpe_annual']:+.2f}/yr, "
           f"P(true Sharpe &gt; 0) = {lk['combined']['psr_vs_zero']:.2f}" if lk.get("combined") else "")
        + "</li>" for lk in looks)
    if not looks:
        look_html = "<p class=muted>Sealed. Run <code>falsealpha unseal</code> once you are done searching.</p>"
    elif len(looks) == 1:
        look_html = f"<ul>{look_rows}</ul>"
    else:
        look_html = (f"<p class=warn>⚠ The holdout has been opened {len(looks)} times. Each extra look is "
                     f"another trial, so the later numbers are optimistic.</p><ul>{look_rows}</ul>")

    s, dates_cfg = cfg["search"], cfg["dates"]
    tile_html = "".join(f"<div class=tile><div class=v>{html.escape(a)}</div><div class=l>{html.escape(b)}</div>"
                        f"<div class=h>{html.escape(c)}</div></div>" for a, b, c in tiles)
    table = "".join(row(i, f) for i, f in enumerate(v["finalists"]))
    return (_TEMPLATE
            .replace("%TITLE%", html.escape(run_dir.name))
            .replace("%SOURCE%", html.escape(cfg["source"]))
            .replace("%META%", html.escape(
                f"{cfg['assets']} assets · train {dates_cfg['train'][0]}–{dates_cfg['train'][1]} · "
                f"validation {dates_cfg['validation'][0]}–{dates_cfg['validation'][1]} · "
                f"holdout {dates_cfg['holdout'][0]}–{dates_cfg['holdout'][1]} · delay {s['delay']} · "
                f"costs {s['cost_bps']} bps/turnover · population {s['population']} × {s['generations']} "
                f"generations · seed {s['seed']} · {cfg['elapsed_seconds']:.1f}s"))
            .replace("%TILES%", tile_html)
            .replace("%TABLE%", table)
            .replace("%LOOKS%", look_html)
            .replace("%LUCK%", f"{v['sr0_train'] * ann:+.2f}")
            .replace("%NTRIALS%", f"{v['n_trials']:,}")
            .replace("%DATA%", json.dumps(payload, allow_nan=False, default=_finite)
                     .replace("</", "<\\/")))


def write(run_dir: str | Path) -> Path:
    out = Path(run_dir) / "report.html"
    out.write_text(build(run_dir))
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>falsealpha · %TITLE%</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
--good:#006300;--bad:#d03b3b;--warn:#8a5a00;--band:rgba(11,11,11,.035)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#0ca30c;--bad:#e66767;--warn:#fab219;--band:rgba(255,255,255,.04)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--good:#0ca30c;--bad:#e66767;--warn:#fab219;--band:rgba(255,255,255,.04)}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:36px 0 4px}p{margin:4px 0}
.muted,.meta{color:var(--ink2)}.meta{font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:20px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px}
.tile .v{font-size:28px;font-weight:650}.tile .l{font-weight:600}.tile .h{color:var(--ink2);font-size:12.5px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;margin-top:10px;
position:relative}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media (max-width:760px){.grid2{grid-template-columns:1fr}}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{fill:var(--muted);font-size:11px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink2);margin-bottom:8px}
.legend i{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:6px;border-radius:1px}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);
border-radius:8px;padding:8px 10px;font-size:12.5px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;
max-width:340px;z-index:2}.tip b{font-variant-numeric:tabular-nums}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.formula code{font-size:12.5px;word-break:break-word}td.why{color:var(--ink2);font-size:12px}
.st{white-space:nowrap;font-weight:600}.st.good{color:var(--good)}.st.bad{color:var(--bad)}
.warn{color:var(--warn);font-weight:600}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
details{margin-top:8px;color:var(--ink2);font-size:13.5px}
</style></head><body><main>
<h1>%TITLE%</h1>
<p class=muted>%SOURCE%</p>
<p class=meta>%META%</p>
<div class=tiles>%TILES%</div>

<h2>Cumulative net return</h2>
<p class=muted>Daily net returns after costs, summed. Shaded: validation, then holdout. The search only saw the train period.</p>
<div class=card id=eq></div>

<div class=grid2>
<div><h2>Trial Sharpe ratios vs the luck bar</h2>
<p class=muted>Train Sharpe of all %NTRIALS% trials, annualised. The line is the best Sharpe expected from that many
<em>unskilled</em> strategies (%LUCK%/yr).</p><div class=card id=hist></div></div>
<div><h2>PBO: in-sample winner's out-of-sample rank</h2>
<p class=muted>One logit per train/validation split (CSCV). Left of 0 the in-sample winner lands in the bottom half
out of sample; the share there is the PBO.</p><div class=card id=pbo></div></div>
</div>

<h2>Search progress</h2>
<p class=muted>Best train Sharpe found so far, by generation.</p>
<div class=card id=prog></div>

<h2>Finalists</h2>
<p class=muted>Pass = train DSR ≥ 0.95 against all trials, validation Sharpe &gt; 0 with DSR ≥ 0.95 against the
finalists, and the SPA gate open. Sharpe ratios annualised.</p>
<div class="card tablewrap"><table><thead><tr><th class=num>#</th><th>formula</th><th class=num>train Sharpe</th>
<th class=num>train DSR</th><th class=num>val Sharpe</th><th class=num>val DSR</th><th>verdict</th><th>why</th></tr>
</thead><tbody>%TABLE%</tbody></table></div>

<h2>Holdout</h2>
<div class=card>%LOOKS%</div>
<details><summary>How to read this</summary>
<p><b>DSR</b> (Deflated Sharpe Ratio, Bailey &amp; López de Prado 2014): probability the true Sharpe beats the best
you'd expect from that many unskilled tries. <b>SPA</b> (Hansen 2005): p-value that the best finalist's validation
returns beat zero after accounting for having looked at all finalists. <b>PBO</b> (Bailey, Borwein, López de Prado
&amp; Zhu 2017): how often picking the in-sample best leads to a below-median strategy out of sample. The holdout
is never shown to the search and opens once; every look is logged.</p></details>
</main>
<script>
const D = %DATA%;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);
  if(parent)parent.appendChild(e);return e}
function ticks(lo,hi,n){const span=hi-lo||1,step0=span/n,mag=Math.pow(10,Math.floor(Math.log10(step0)));
  const step=[1,2,5,10].map(m=>m*mag).find(s=>s>=step0);const out=[];for(let v=Math.ceil(lo/step)*step;v<=hi+1e-12;v+=step)out.push(+v.toFixed(10));return out}
function frame(host,h){const W=host.clientWidth-32||600,m={l:46,r:12,t:10,b:26};
  const svg=el("svg",{viewBox:`0 0 ${W} ${h}`,role:"img"});host.appendChild(svg);
  const tip=document.createElement("div");tip.className="tip";host.appendChild(tip);
  return {svg,tip,W,H:h,m,iw:W-m.l-m.r,ih:h-m.t-m.b}}
function yAxis(f,lo,hi,fmt){const y=v=>f.m.t+f.ih*(1-(v-lo)/(hi-lo||1));
  for(const v of ticks(lo,hi,5)){el("line",{x1:f.m.l,x2:f.W-f.m.r,y1:y(v),y2:y(v),stroke:css("--grid"),"stroke-width":1},f.svg);
    const t=el("text",{x:f.m.l-6,y:y(v)+4,"text-anchor":"end"},f.svg);t.textContent=fmt(v)}return y}
function showTip(f,html,px,py){f.tip.innerHTML=html;f.tip.style.display="block";
  const r=f.tip.getBoundingClientRect(),W=f.svg.parentNode.clientWidth;
  f.tip.style.left=Math.min(Math.max(8,px+14),W-r.width-8)+"px";f.tip.style.top=Math.max(8,py-r.height-8)+"px"}

function bars(id,counts,edges,refs,fmt){const host=document.getElementById(id);
  if(!counts.length){host.textContent="No data.";return}
  const f=frame(host,220),lo=edges[0],hi=edges[edges.length-1],max=Math.max(...counts);
  const x=v=>f.m.l+f.iw*(v-lo)/(hi-lo||1),y=yAxis(f,0,max,v=>v);
  counts.forEach((c,i)=>{const x0=x(edges[i])+1,x1=x(edges[i+1])-1,top=y(c),base=y(0);
    if(c>0){const r=Math.min(4,(x1-x0)/2,base-top);
      el("path",{d:`M${x0},${base}V${top+r}Q${x0},${top} ${x0+r},${top}H${x1-r}Q${x1},${top} ${x1},${top+r}V${base}Z`,fill:css("--s1")},f.svg)}
    const hit=el("rect",{x:x(edges[i]),y:f.m.t,width:Math.max(1,x(edges[i+1])-x(edges[i])),height:f.ih,fill:"transparent"},f.svg);
    hit.addEventListener("pointermove",e=>{const b=host.getBoundingClientRect();
      showTip(f,`${fmt(edges[i])} to ${fmt(edges[i+1])}: <b>${c}</b>`,e.clientX-b.left,e.clientY-b.top)});
    hit.addEventListener("pointerleave",()=>f.tip.style.display="none")});
  el("line",{x1:f.m.l,x2:f.W-f.m.r,y1:y(0),y2:y(0),stroke:css("--axis")},f.svg);
  for(const v of ticks(lo,hi,6)){const t=el("text",{x:x(v),y:f.H-6,"text-anchor":"middle"},f.svg);t.textContent=fmt(v)}
  for(const r of refs){if(r.at==null||r.at<lo||r.at>hi)continue;
    el("line",{x1:x(r.at),x2:x(r.at),y1:f.m.t,y2:y(0),stroke:css("--ink"),"stroke-width":1.5,"stroke-dasharray":"4 3"},f.svg);
    const t=el("text",{x:x(r.at)+5,y:f.m.t+10},f.svg);t.textContent=r.label;t.style.fill=css("--ink2")}}

function lines(id,xs,series,opts){const host=document.getElementById(id);
  if(!series.length){host.textContent="No finalists.";return}
  const colors=["--s1","--s2","--s3"].map(css);
  if(series.length>1){const lg=document.createElement("div");lg.className="legend";
    series.forEach((s,i)=>{const it=document.createElement("span");it.innerHTML=`<i style="background:${colors[i]}"></i>`;
      it.appendChild(document.createTextNode(s.name));lg.appendChild(it)});host.appendChild(lg)}
  const f=frame(host,opts.h||260),n=xs.length;let lo=Infinity,hi=-Infinity;
  for(const s of series)for(const v of s.values)if(v!=null){lo=Math.min(lo,v);hi=Math.max(hi,v)}
  if(!isFinite(lo)){lo=0;hi=1}if(opts.zero){lo=Math.min(lo,0);hi=Math.max(hi,0)}const pad=(hi-lo)*.05||1;lo-=pad;hi+=pad;
  const x=i=>f.m.l+f.iw*i/Math.max(1,n-1),y=yAxis(f,lo,hi,opts.fmt);
  for(const b of opts.bands||[]){el("rect",{x:x(b.from),y:f.m.t,width:x(b.to)-x(b.from),height:f.ih,fill:css("--band")},f.svg);
    const t=el("text",{x:x(b.from)+6,y:f.m.t+12},f.svg);t.textContent=b.label}
  if(opts.zero)el("line",{x1:f.m.l,x2:f.W-f.m.r,y1:y(0),y2:y(0),stroke:css("--axis")},f.svg);
  series.forEach((s,i)=>{let d="",pen=false;s.values.forEach((v,j)=>{if(v==null){pen=false;return}
      d+=(pen?"L":"M")+x(j).toFixed(1)+","+y(v).toFixed(1);pen=true});
    el("path",{d,fill:"none",stroke:colors[i],"stroke-width":2,"stroke-linejoin":"round"},f.svg)});
  const step=Math.max(1,Math.round(n/6));for(let i=0;i<n;i+=step){const t=el("text",{x:x(i),y:f.H-6,"text-anchor":"middle"},f.svg);t.textContent=opts.xfmt(xs[i])}
  const cross=el("line",{y1:f.m.t,y2:f.m.t+f.ih,stroke:css("--axis"),"stroke-width":1,visibility:"hidden"},f.svg);
  const dots=series.map((s,i)=>el("circle",{r:4,fill:colors[i],stroke:css("--surface"),"stroke-width":2,visibility:"hidden"},f.svg));
  const hit=el("rect",{x:f.m.l,y:f.m.t,width:f.iw,height:f.ih,fill:"transparent"},f.svg);
  hit.addEventListener("pointermove",e=>{const b=f.svg.getBoundingClientRect(),sx=(e.clientX-b.left)*f.W/b.width;
    const j=Math.max(0,Math.min(n-1,Math.round((sx-f.m.l)/f.iw*(n-1))));cross.setAttribute("x1",x(j));cross.setAttribute("x2",x(j));
    cross.setAttribute("visibility","visible");let rows="";series.forEach((s,i)=>{const v=s.values[j];
      if(v==null){dots[i].setAttribute("visibility","hidden");return}dots[i].setAttribute("cx",x(j));dots[i].setAttribute("cy",y(v));
      dots[i].setAttribute("visibility","visible");rows+=`<div><i style="display:inline-block;width:12px;height:2px;background:${colors[i]};vertical-align:middle;margin-right:6px"></i>${s.name.length>48?s.name.slice(0,47)+"…":s.name}: <b>${opts.fmt(v)}</b></div>`});
    const h=host.getBoundingClientRect();showTip(f,`<div style="margin-bottom:4px">${opts.xfmt(xs[j])}</div>`+rows,e.clientX-h.left,e.clientY-h.top)});
  hit.addEventListener("pointerleave",()=>{f.tip.style.display="none";cross.setAttribute("visibility","hidden");dots.forEach(d=>d.setAttribute("visibility","hidden"))})}

function draw(){for(const id of ["eq","hist","pbo","prog"])document.getElementById(id).replaceChildren();
  const pct=v=>(v*100).toFixed(0)+"%";
  lines("eq",D.dates,D.curves,{fmt:pct,xfmt:s=>s.slice(0,7),zero:true,
    bands:[{from:D.split.val,to:D.split.hold,label:"validation"},{from:D.split.hold,to:D.dates.length-1,label:"holdout"}]});
  bars("hist",D.hist.counts,D.hist.edges,[{at:D.hist.luck,label:"luck bar"}],v=>v.toFixed(1));
  if(D.pbo.length){const lo=Math.min(...D.pbo,-0.5),hi=Math.max(...D.pbo,0.5),k=24,e=[...Array(k+1)].map((_,i)=>lo+(hi-lo)*i/k),
    c=Array(k).fill(0);for(const v of D.pbo)c[Math.min(k-1,Math.floor((v-lo)/(hi-lo)*k))]++;
    bars("pbo",c,e,[{at:0,label:"0"}],v=>v.toFixed(1))}else document.getElementById("pbo").textContent="Not enough finalists.";
  lines("prog",D.progress.map(p=>p[0]),[{name:"best train Sharpe",values:D.progress.map(p=>p[1])}],
    {fmt:v=>v.toFixed(2),xfmt:g=>"gen "+g,h:180})}
draw();let rt;addEventListener("resize",()=>{clearTimeout(rt);rt=setTimeout(draw,150)});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);
</script></body></html>
"""
