"""Assembling the standalone explorer: payload + views -> one self-contained file.

SELF-CONTAINED IS A REQUIREMENT, NOT A PREFERENCE. The file must open from disk, with no network,
forever. Every deliverable this project has shipped that depended on a content-delivery network has
eventually stopped rendering something -- one of them still needs the network for its equations.
So: no external script, no external stylesheet, no remote font, no embedded remote image.

THE RATIO IS THE DESIGN. The predecessor's explorer is 98.4% data and 1.6% view code, which is
exactly why a new view is cheap there. This keeps that shape: the payload dominates, the view layer
is small, and both are emitted separately so a view can be rebuilt without regenerating anything.

PALETTE. The six arms are a categorical encoding, so hues are assigned in FIXED slot order and
never cycled -- colour follows the arm, not its rank, so filtering to three arms does not repaint
the survivors. The values are the reference palette's slots 1-6, used unmodified.

  NOTE, recorded rather than glossed: the palette validator is a Node script and this machine has
  no `node`, so the six-hue set was NOT re-validated here. It is used exactly as shipped, in the
  documented fixed order, which is the configuration the reference validated. Anyone changing a hue
  must re-run `scripts/validate_palette.js` before shipping it.
"""
from __future__ import annotations

import json
import os

from .registry import all_views, check_requirements

#: Reference palette, slots 1-6, light and dark. Dark is the same hues stepped for the dark
#: surface -- a selected set, not an automatic flip.
ARM_COLOURS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ARM_COLOURS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
ARM_ORDER = ["pure", "hazard", "window", "het", "censor", "realfull"]

_CSS = """
.viz-root{color-scheme:light;
 --surface-1:#fcfcfb;--surface-2:#f2f2f0;--line:#d8d8d4;
 --text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#77766f;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
 color-scheme:dark;--surface-1:#1a1a19;--surface-2:#232321;--line:#3a3a37;
 --text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#94938a;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
 --surface-1:#1a1a19;--surface-2:#232321;--line:#3a3a37;
 --text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#94938a;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--surface-1);color:var(--text-primary);
 font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
h1{font-size:19px;margin:0 0 2px}
.sub{color:var(--text-secondary);font-size:12.5px;margin:0 0 14px}
.prov{background:var(--surface-2);border:1px solid var(--line);border-radius:8px;
 padding:10px 12px;margin-bottom:14px;font-size:12.5px;color:var(--text-secondary)}
.prov b{color:var(--text-primary)}
.warn{border-left:3px solid var(--s2);padding-left:9px;margin-top:8px}
.ctl{display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:10px 12px;
 border:1px solid var(--line);border-radius:8px;background:var(--surface-2);margin-bottom:14px}
label{font-size:12.5px;color:var(--text-secondary);margin-right:6px}
select,button{font:inherit;font-size:13px;padding:3px 7px;background:var(--surface-1);
 color:var(--text-primary);border:1px solid var(--line);border-radius:6px}
.views{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr))}
.card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--surface-1)}
.card h2{font-size:14.5px;margin:0 0 2px}
.card .desc{font-size:12px;color:var(--text-muted);margin:0 0 10px}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--text-secondary);
 margin-top:8px}
.legend i{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:4px;
 vertical-align:-1px}
.denom{font-size:11.5px;color:var(--text-muted);margin-top:6px}
.filternote{font-size:12.5px;color:var(--text-secondary);margin:0 0 12px;padding:7px 10px;
 border-radius:7px;background:var(--surface-2);border:1px solid var(--line)}
.filternote.on{border-left:3px solid var(--s1)}
table{border-collapse:collapse;font-size:12px;width:100%;margin-top:8px}
.scrollx{overflow-x:auto;max-width:100%}
.scrollx table{min-width:640px;white-space:nowrap}
th,td{text-align:right;padding:3px 6px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--line);
 border-radius:6px;padding:5px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.18);
 opacity:0;transition:opacity .08s}
svg{display:block;width:100%;height:auto}
.grid{stroke:var(--line);stroke-width:1}
.axis{stroke:var(--line);stroke-width:1}
.axlab{fill:var(--text-muted);font-size:10.5px}
.tick{stroke:var(--line);stroke-width:1}
.ticklab{fill:var(--text-muted);font-size:9.5px}
.gridline{stroke:var(--line);stroke-width:1;stroke-opacity:.35}
.nominal{stroke:var(--s6);stroke-width:1.25;stroke-dasharray:4 3}
details.arms{background:var(--surface-2);border:1px solid var(--line);border-radius:8px;
 padding:8px 12px;margin-bottom:14px;font-size:12.5px;color:var(--text-secondary)}
details.arms summary{cursor:pointer;color:var(--text-primary);font-weight:600}
details.arms table{margin-top:10px}
details.arms td,details.arms th{text-align:left;vertical-align:top}
.state{font-weight:600;white-space:nowrap}
.state-invisible{color:var(--s2)}
.state-undeclared{color:var(--s4)}
.state-handled{color:var(--s6)}
.state-none,.state-mixed,.state-unknown{color:var(--text-muted)}
"""

_JS_HELPERS = r"""
const ARMS = %ARMS%;
function armColour(a){const i=ARMS.indexOf(a);return `var(--s${(i<0?0:i)+1})`;}
const el=(t,a={},...k)=>{const n=document.createElementNS(
  t==='svg'||['g','rect','circle','line','path','text','title'].includes(t)
  ?'http://www.w3.org/2000/svg':'http://www.w3.org/1999/xhtml',t);
  for(const[p,v] of Object.entries(a))n.setAttribute(p,v);
  for(const c of k)n.appendChild(typeof c==='string'?document.createTextNode(c):c);return n;};

let TIP;
function tipInit(){TIP=document.createElement('div');TIP.className='tip';document.body.appendChild(TIP);}
function hoverable(node,text){
  node.addEventListener('pointerenter',e=>{TIP.textContent=text;TIP.style.opacity=1;});
  node.addEventListener('pointermove',e=>{TIP.style.left=(e.clientX+12)+'px';
    TIP.style.top=(e.clientY+12)+'px';});
  node.addEventListener('pointerleave',()=>{TIP.style.opacity=0;});
  return node;}

function legendFor(arms){
  const d=document.createElement('div');d.className='legend';
  arms.forEach(a=>{const s=document.createElement('span');
    s.innerHTML=`<i style="background:${armColour(a)}"></i>${a}`;d.appendChild(s);});
  return d;}

function denomLine(txt){const d=document.createElement('div');d.className='denom';
  d.textContent=txt;return d;}

// --- axis ticks --------------------------------------------------------------------------
// Every panel drew an axis LINE and no numbers, so a reader could see a shape but not read a
// value off it: whether a histogram bar was 5 cohorts or 500, whether a cost was 300 s or 30,000.
// One helper, so the three call sites cannot drift apart.

function niceTicks(lo,hi,target){
  // 1/2/5 x 10^k steps -- the values a reader can do arithmetic on.
  target=target||5;
  if(!isFinite(lo)||!isFinite(hi))return [];
  if(hi<=lo)return [lo];                       // zero span: one tick, not a division by zero
  const raw=(hi-lo)/target, mag=Math.pow(10,Math.floor(Math.log10(raw))), n=raw/mag;
  const step=mag*(n<1.5?1:n<3?2:n<7?5:10);
  const out=[]; let t=Math.ceil(lo/step)*step;
  // Guard the loop on count as well as value: a pathological step would otherwise hang the page.
  for(let i=0;i<200&&t<=hi+step*1e-9;i++,t+=step)out.push(Math.abs(t)<step*1e-9?0:t);
  return out;}

function fmtNum(v){
  if(v===null||v===undefined||!isFinite(v))return '';
  const a=Math.abs(v);
  if(a===0)return '0';
  if(a>=1e5)return (v/1e3).toFixed(0)+'k';
  if(a>=1e3)return (v/1e3).toFixed(a>=1e4?0:1)+'k';
  if(a>=100)return v.toFixed(0);
  if(a>=10)return v.toFixed(1);
  if(a>=1)return v.toFixed(2);
  if(a>=0.01)return v.toFixed(2);
  return v.toExponential(0);}

function axes(svg,o){
  // o: {x0,x1,y0,y1,sx,sy,W,H,P,xlab,ylab,xfmt,yfmt,grid}
  const xf=o.xfmt||fmtNum, yf=o.yfmt||fmtNum, P=o.P, W=o.W, H=o.H;
  const xt=niceTicks(o.x0,o.x1,o.xticks||5), yt=niceTicks(o.y0,o.y1,o.yticks||4);
  svg.appendChild(el('line',{x1:P,y1:H-P,x2:W-8,y2:H-P,class:'axis'}));
  svg.appendChild(el('line',{x1:P,y1:10,x2:P,y2:H-P,class:'axis'}));
  xt.forEach(t=>{const x=o.sx(t); if(x<P-0.5||x>W-6)return;
    svg.appendChild(el('line',{x1:x,y1:H-P,x2:x,y2:H-P+4,class:'tick'}));
    svg.appendChild(el('text',{x:x,y:H-P+14,'text-anchor':'middle',class:'ticklab'},xf(t)));});
  yt.forEach(t=>{const y=o.sy(t); if(y<8||y>H-P+0.5)return;
    svg.appendChild(el('line',{x1:P-4,y1:y,x2:P,y2:y,class:'tick'}));
    if(o.grid!==false)svg.appendChild(el('line',{x1:P,y1:y,x2:W-8,y2:y,class:'gridline'}));
    svg.appendChild(el('text',{x:P-6,y:y+3,'text-anchor':'end',class:'ticklab'},yf(t)));});
  if(o.xlab)svg.appendChild(el('text',{x:(W+P)/2,y:H-6,'text-anchor':'middle',class:'axlab'},
    o.xlab));
  if(o.ylab){const t=el('text',{x:0,y:0,'text-anchor':'middle',class:'axlab',
    transform:`translate(11,${(H-P+10)/2}) rotate(-90)`},o.ylab);svg.appendChild(t);}
  return svg;}

// --- bars, with the denominator and decline rate ALWAYS drawn beside them ----------------
function barsWithDenominator(rows,q,level){
  // THE NOMINAL LINE. "0.823" is only a number until you can see what it is coverage OF, and
  // under-coverage should be visible without arithmetic. The level is read from the claims
  // (payload provenance.level), never assumed.
  const W=440,H=42+rows.length*26,L=88,BW=W-L-116;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  const nx=(level===null||level===undefined)?null:L+BW*level;
  if(nx!==null){
    svg.appendChild(el('line',{x1:nx,y1:6,x2:nx,y2:H-24,class:'nominal'}));
    svg.appendChild(el('text',{x:nx,y:H-12,'text-anchor':'middle',class:'ticklab'},
                      `nominal ${level}`));
  }
  rows.forEach((r,i)=>{
    const y=12+i*26;
    svg.appendChild(el('text',{x:L-8,y:y+11,'text-anchor':'end',class:'axlab'},r.arm));
    const w=r.cov===null?0:Math.max(2,BW*r.cov);
    const bar=el('rect',{x:L,y:y,width:w,height:14,rx:4,fill:armColour(r.arm)});
    hoverable(bar,`${r.arm}: coverage ${r.cov===null?'n/a':r.cov.toFixed(3)} `+
      `of the ${level===null||level===undefined?'?':level*100+'%'} interval `+
      `(nominal ${level===null||level===undefined?'?':level}) `+
      `on ${r.claimed} claims; declined ${r.declined}/${r.total}`);
    svg.appendChild(bar);
    svg.appendChild(el('text',{x:L+w+8,y:y+11,class:'axlab'},
      (r.cov===null?'n/a':r.cov.toFixed(3))+`  n=${r.claimed}  decl ${(r.declined/r.total).toFixed(2)}`));
  });
  const box=document.createElement('div');box.appendChild(svg);
  box.appendChild(legendFor(rows.map(r=>r.arm)));
  const lv=(level===null||level===undefined)?'the claimed interval'
                                            :`the ${(level*100).toFixed(0)}% interval`;
  box.appendChild(denomLine(
    `${q} · fraction of cohorts whose ${lv} contained the truth`+
    `${level===null||level===undefined?'':` (nominal ${level})`} · two-sided claims only · `+
    `n = claims made · decl = decline rate over all cohorts · arms scored separately, never pooled`));
  return box;}

function scatter(pts,xlab,ylab){
  // EMPTY GUARD. Math.min(...[]) is Infinity, so an empty selection used to produce NaN
  // coordinates and render as an axis-only box -- indistinguishable from "this method had no
  // results". Say which it is.
  if(!pts.length)return denomLine('no cohort in this selection has a value for this quantity');
  const W=440,H=268,P=48;
  const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=0,y1=Math.max(...ys,1e-9);
  if(x1<=x0){x0-=0.5;x1+=0.5;}                 // a single distinct x still needs a readable axis
  const sx=v=>P+(W-P-10)*((v-x0)/((x1-x0)||1));
  const sy=v=>H-P-(H-P-16)*((v-y0)/((y1-y0)||1));
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  axes(svg,{x0,x1,y0,y1,sx,sy,W,H,P,xlab,ylab});
  pts.forEach(p=>{const c=el('circle',{cx:sx(p.x),cy:sy(p.y),r:3.2,
    fill:armColour(p.arm),'fill-opacity':.8,stroke:'var(--surface-1)','stroke-width':1.5});
    hoverable(c,`${p.arm} · x=${p.x.toFixed(3)} · y=${p.y.toFixed(3)}`);svg.appendChild(c);});
  const box=document.createElement('div');box.appendChild(svg);
  box.appendChild(legendFor([...new Set(pts.map(p=>p.arm))]));
  box.appendChild(denomLine(`${pts.length} cohorts with a value`));
  return box;}


function refCompare(pts,xlab,ylab){
  // Method cost against the REFERENCE's cost, one point per cohort, with a y=x diagonal.
  // Cost for this workload is flat in sample size (the Fokker-Planck grid comes from the lifespan
  // scale, not the number of individuals), so plotting seconds against n showed stripes and no
  // signal. Against the reference, the diagonal is the question actually being asked: above it is
  // slower, below is faster, and a method that IS the reference lies exactly on it.
  if(!pts.length)return denomLine('no cohort has both a method and a reference timing');
  const W=440,H=276,P=52;
  const box=document.createElement('div');
  // LOG BY DEFAULT WHEN THE RANGE DEMANDS IT. Measured on the full tier the cost runs 335 s to
  // 69,410 s (one E. coli variant whose recording resolution forces ~63,000 solver steps), so on
  // a linear axis every other cohort is squashed into the bottom-left corner. The toggle is
  // offered rather than forced, because a linear axis reads the 2x ratios more directly.
  const vals=pts.flatMap(p=>[p.x,p.y]).filter(v=>v>0);
  const spread=vals.length?Math.max(...vals)/Math.min(...vals):1;
  if(box._log===undefined)box._log=spread>50;
  const draw=()=>{
    const useLog=box._log;
    const f=v=>useLog?Math.log10(Math.max(v,1e-9)):v;
    // Domain from the REAL values only. A 1e-9 sentinel in this list put the log axis floor at
    // -9 dex, so ticks ran 1e-8..10k for data that never leaves 335 s..69,410 s and every point
    // piled into the top-right corner.
    const pos=pts.flatMap(p=>[p.x,p.y]).filter(v=>isFinite(v)&&v>0);
    let lo,hi;
    if(useLog){ lo=f(Math.min(...pos)); hi=f(Math.max(...pos));
                const pad=Math.max(0.05,(hi-lo)*0.04); lo-=pad; hi+=pad; }
    else      { lo=0; hi=Math.max(...pts.flatMap(p=>[p.x,p.y]),1e-9); }
    if(hi<=lo)hi=lo+1;
    const sx=v=>P+(W-P-10)*((f(v)-lo)/((hi-lo)||1));
    const sy=v=>H-P-(H-P-16)*((f(v)-lo)/((hi-lo)||1));
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
    const fmt=useLog?(t=>fmtNum(Math.pow(10,t))):fmtNum;
    axes(svg,{x0:lo,x1:hi,y0:lo,y1:hi,sx:t=>P+(W-P-10)*((t-lo)/((hi-lo)||1)),
              sy:t=>H-P-(H-P-16)*((t-lo)/((hi-lo)||1)),
              W,H,P,xlab,ylab,xfmt:fmt,yfmt:fmt});
    // y=x in DATA space, so it stays the equal-cost line under either scaling.
    const d0=useLog?Math.pow(10,lo):0, d1=useLog?Math.pow(10,hi):hi;
    svg.appendChild(el('line',{x1:sx(d0),y1:sy(d0),x2:sx(d1),y2:sy(d1),class:'axis',
                               'stroke-dasharray':'4 3','stroke-opacity':.65}));
    svg.appendChild(el('text',{x:sx(d1)-4,y:sy(d1)-5,'text-anchor':'end',class:'axlab'},
                      'equal cost'));
    pts.forEach(p=>{const c=el('circle',{cx:sx(p.x),cy:sy(p.y),r:3.2,
      fill:armColour(p.arm),'fill-opacity':.8,stroke:'var(--surface-1)','stroke-width':1.5});
      const r=p.x>0?(p.y/p.x):null;
      hoverable(c,`${p.arm} · method ${p.y.toFixed(0)}s · reference ${p.x.toFixed(0)}s`+
                  (r!==null?` · ${r.toFixed(2)}x`:''));
      svg.appendChild(c);});
    return svg;};
  let svgNode=draw();box.appendChild(svgNode);
  const tog=document.createElement('button');
  tog.textContent=box._log?'linear axes':'log axes';
  tog.onclick=()=>{box._log=!box._log;tog.textContent=box._log?'linear axes':'log axes';
    const n=draw();box.replaceChild(n,svgNode);svgNode=n;};
  box.appendChild(tog);
  box.appendChild(legendFor([...new Set(pts.map(p=>p.arm))]));
  const ratios=pts.filter(p=>p.x>0).map(p=>p.y/p.x).sort((a,b)=>a-b);
  let note=`${pts.length} cohorts with both timings`;
  if(ratios.length){
    const med=ratios[Math.floor(ratios.length/2)];
    const slower=ratios.filter(r=>r>1).length;
    note+=` · median ${med.toFixed(2)}x the reference · slower on ${slower}/${ratios.length}`;
  }
  box.appendChild(denomLine(note));
  return box;}

function histogram(vals,xlab,q){
  // A bar height was unreadable without a counted y axis: 5 cohorts and 500 looked the same.
  const W=440,H=216,P=46,NB=18;
  const box=document.createElement('div');
  if(!vals.length){box.appendChild(denomLine('no closed claims to show'));return box;}
  const lo=Math.min(...vals),hi=Math.max(...vals);
  const bins=new Array(NB).fill(0);
  vals.forEach(v=>{let i=Math.floor(NB*(v-lo)/((hi-lo)||1));if(i>=NB)i=NB-1;bins[i]++;});
  const mx=Math.max(...bins);
  const sx=v=>P+(W-P-10)*((v-lo)/((hi-lo)||1));
  const sy=c=>H-P-(H-P-16)*(c/(mx||1));
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  // Axes first so the faint gridlines sit BEHIND the bars rather than across them.
  axes(svg,{x0:lo,x1:hi,y0:0,y1:mx,sx,sy,W,H,P,xlab,ylab:'cohorts',
            yfmt:v=>String(Math.round(v))});
  bins.forEach((c,i)=>{const w=(W-P-10)/NB,x=P+i*w,h=(H-P)-sy(c);
    const r=el('rect',{x:x+1,y:sy(c),width:w-2,height:h,rx:3,fill:'var(--s1)'});
    hoverable(r,`${c} cohorts in [${(lo+(hi-lo)*i/NB).toFixed(2)}, `+
      `${(lo+(hi-lo)*(i+1)/NB).toFixed(2)}) dex`);svg.appendChild(r);});
  box.appendChild(svg);
  const med=[...vals].sort((a,b)=>a-b)[Math.floor(vals.length/2)];
  box.appendChild(denomLine(`${q} · ${vals.length} CLOSED claims · median ${med.toFixed(3)} dex · `+
    `one-sided claims have no width and are excluded`));
  return box;}

function stackedShapes(rows,q){
  const W=440,H=30+rows.length*26,L=88,BW=W-L-70;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  const keys=[['closed','var(--s1)'],['upper','var(--s3)'],['lower','var(--s4)'],
              ['none','var(--text-muted)']];
  rows.forEach((r,i)=>{
    const y=12+i*26,tot=r.none+r.upper+r.lower+r.closed||1;let x=L;
    svg.appendChild(el('text',{x:L-8,y:y+11,'text-anchor':'end',class:'axlab'},r.arm));
    keys.forEach(([k,c])=>{const w=BW*(r[k]/tot);if(w<=0)return;
      const rect=el('rect',{x:x+1,y:y,width:Math.max(0,w-2),height:14,rx:3,fill:c});
      hoverable(rect,`${r.arm} · ${k}: ${r[k]} of ${tot}`);svg.appendChild(rect);x+=w;});
  });
  const box=document.createElement('div');box.appendChild(svg);
  const lg=document.createElement('div');lg.className='legend';
  keys.forEach(([k,c])=>{const s=document.createElement('span');
    s.innerHTML=`<i style="background:${c}"></i>${k}`;lg.appendChild(s);});
  box.appendChild(lg);
  box.appendChild(denomLine(`${q} · shapes are never pooled into one coverage`));
  return box;}

// --- ported from the predecessor explorer -------------------------------------------------
// The six surfaces of `mle_report/explorer.html`, rebuilt on this payload. Two of its rules are
// kept deliberately, because each was written against a real mistake:
//   * "one marker per PARAMETER POINT, not per cohort" -- seed replicates must not make a thin
//     design look dense;
//   * grey is not red -- "declined" and "wrong" are different statements.

function byPoint(idx){
  // Group cohort indices by their parameter point (cluster_key).
  const m=new Map();
  idx.forEach(i=>{const k=D.cohort.cluster_key[i];
    if(!m.has(k))m.set(k,[]);m.get(k).push(i);});
  return m;}

function median(v){const s=v.filter(x=>x!==null&&isFinite(x)).sort((a,b)=>a-b);
  return s.length?s[Math.floor(s.length/2)]:null;}

function clusterRobustSE(values,clusters){
  // Sandwich SE, clustered on the parameter point. Seed replicates of one truth are not
  // independent draws, and treating them as such understates the SE.
  if(!values.length)return null;
  const mean=values.reduce((a,b)=>a+b,0)/values.length;
  const g=new Map();
  values.forEach((v,i)=>{const k=clusters[i];
    g.set(k,(g.get(k)||0)+(v-mean));});
  const G=g.size; if(G<2)return null;
  let ss=0; g.forEach(s=>{ss+=s*s;});
  const corr=G/(G-1);
  return Math.sqrt(corr*ss)/values.length;}

function accColour(e){return e===null?'var(--text-muted)'
  :e<0.2?'var(--s6)':e<0.5?'var(--s4)':'var(--s2)';}
function covColour(c){return c===null?'var(--text-muted)'
  :c>=0.90?'var(--s6)':c>=0.80?'var(--s4)':'var(--s2)';}

// --- combinations: how tightly the data pin them, and the error at the fitted point ----------
// A combination (beta/eta, beta/(kappa+1), ...) has no interval. Its error used to be built from
// claim POINTS, so it was grey wherever an input was declined -- kappa on most cohorts. Two
// readings are offered under the "combinations" toggle, and neither is a claim:
//   resolvability (default)  the spread of the combination over the fit's own near-optimal points
//     (payload `result.spread`, a LOWER bound on what the data leave undetermined). The error is
//     drawn only where the data pin the combination; elsewhere the point says so instead.
//   error at fitted theta     the error with each declined input filled from the fitted point
//     (`result.err_theta`), drawn as a RING where a fill was used, with the fit's own staleness
//     class and any box-edge parameter in the hover.
// A method that stores no near-optimal record has no spread; it keeps today's claim-point error.
function isCombo(q){return (D.accuracy_only_quantities||[]).indexOf(q)>=0;}
function comboMode(){const e=document.getElementById('cmode');return e?e.value:'resolve';}
function CR(){return D.combo_resolution||{};}   // D is defined after these helpers
function resClass(s){
  if(s===null||s===undefined)return 'unknown';
  return s<=CR().resolved_dex?'resolved':s<=CR().unresolved_dex?'partly':'unresolved';}
function errAt(qi,i){
  // THE error every view reads for cohort i. For a parameter it is exactly `result.err`.
  const q=D.quantities[qi];
  if(!isCombo(q)||!D.result.err_theta)return D.result.err[qi][i];
  if(comboMode()==='theta')return D.result.err_theta[qi][i];
  const c=resClass(D.result.spread[qi][i]);
  return c==='resolved'?D.result.err_theta[qi][i]:c==='unknown'?D.result.err[qi][i]:null;}
function clampedAt(qi,i){
  // The floor flag of WHICHEVER error errAt returned, so "unclamped" subsets stay honest.
  const q=D.quantities[qi];
  if(!isCombo(q)||!D.result.clamped_theta)return D.result.clamped[qi][i];
  if(comboMode()==='theta')return D.result.clamped_theta[qi][i]||0;
  const c=resClass(D.result.spread[qi][i]);
  return c==='resolved'?(D.result.clamped_theta[qi][i]||0):c==='unknown'?D.result.clamped[qi][i]:0;}
function comboCounts(qi,idx){
  const k={resolved:0,partly:0,unresolved:0,unknown:0,filled:0,stale:0,edge:0,n:idx.length};
  idx.forEach(i=>{k[resClass(D.result.spread[qi][i])]++;
    if(D.result.theta_fill&&D.result.theta_fill[qi][i]===1){k.filled++;
      if(D.cohort.fit_stale&&D.cohort.fit_stale[i]&&D.cohort.fit_stale[i]!=='none')k.stale++;
      if(D.cohort.theta_at_edge&&(D.cohort.theta_at_edge[i]||[]).length)k.edge++;}});
  return k;}
function comboNote(qi,idx){
  // One line stating what a combination figure is drawn from, with its counts.
  if(!isCombo(D.quantities[qi])||!D.result.spread)return '';
  const k=comboCounts(qi,idx);
  if(comboMode()==='theta')
    return `NOT AN ACCURACY OF A CLAIM: the method claims nothing on a combination; this is the error `+
      `with each declined input taken from the fit. ${k.filled} of ${k.n} cohorts had an input filled `+
      `(dark outline), of which ${k.stale} carry a stale fitted point and ${k.edge} a parameter on a box `+
      `edge. ${k.partly+k.unresolved} of the ${k.n-k.unknown} cohorts with a near-optimal record are NOT `+
      `pinned to ${CR().resolved_dex} dex by their own fits -- switch to resolvability to see which.`;
  return `Resolvability: the data pin this combination to within ${CR().resolved_dex} dex on `+
    `${k.resolved} of ${k.n} cohorts (error shown); ${k.partly} partly (${CR().resolved_dex}-`+
    `${CR().unresolved_dex} dex) and ${k.unresolved} unresolved (> ${CR().unresolved_dex} dex) are drawn `+
    `as rings, their spread in the hover` + (k.unknown?`; ${k.unknown} carry no near-optimal record `+
    `(claim-point error)`:'')+`. The spread is a LOWER bound on what the data leave undetermined, not an `+
    `interval and not an accuracy. Markers are parameter points (median over seed replicates); these `+
    `counts are cohorts.`;}
function pointStyle(qi,ii){
  // How one PARAMETER POINT (seed replicates ii) is drawn in the accuracy views.
  const q=D.quantities[qi];
  const v=median(ii.map(i=>errAt(qi,i)));
  const base={fill:accColour(v),stroke:'var(--surface-1)',ring:false,dash:false,v,extra:''};
  if(!isCombo(q)||!D.result.spread)return base;
  const sp=median(ii.map(i=>D.result.spread[qi][i]));
  const nf=ii.filter(i=>D.result.theta_fill[qi][i]===1).length;
  const st=ii.filter(i=>D.cohort.fit_stale&&D.cohort.fit_stale[i]&&D.cohort.fit_stale[i]!=='none').length;
  const ed=[...new Set(ii.flatMap(i=>(D.cohort.theta_at_edge||[])[i]||[]))];
  const extra=` · spread ${sp===null?'not measurable':sp.toFixed(2)+' dex'}`+
    (nf?` · ${nf}/${ii.length} filled from the fitted point`:'')+
    (st?` · ${st} stale fit(s)`:'')+(ed.length?` · on a box edge: ${ed.join(', ')}`:'');
  if(comboMode()==='theta')
    return Object.assign(base,{halo:nf>0,stroke:nf>0?'var(--text-primary)':base.stroke,extra});
  const c=resClass(sp);
  if(c==='unresolved')
    return {fill:'none',stroke:'var(--text-muted)',ring:true,v:null,extra,cls:c};
  if(c==='partly')
    return {fill:'var(--text-muted)',fillOpacity:.35,stroke:'var(--text-muted)',ring:false,halfTone:true,
            v:null,extra,cls:c};
  return Object.assign(base,{extra});}
function accLegendKeys(qi){
  // [label, swatch style] -- rings, half-tones and outlines, not new hues: these are STATES, not
  // severities.
  const keys=[['|err| &lt; 0.2 dex','background:var(--s6)'],['0.2-0.5 dex','background:var(--s4)'],
              ['&gt; 0.5 dex','background:var(--s2)']];
  if(!isCombo(D.quantities[qi])||!D.result.spread)
    return keys.concat([['no error','background:var(--text-muted)']]);
  if(comboMode()==='theta')
    return [['NOT an accuracy of a claim -- error with declined inputs taken from the fit:','display:none']]
      .concat(keys).concat([['dark outline: an input filled from the fitted point',
                         'background:var(--surface-2);border:2px solid var(--text-primary)'],
                        ['no error','background:var(--text-muted)']]);
  return [[`pinned = the fit's own near-optimal points agree within ${CR().resolved_dex} dex `+
           `(not an accuracy); error shown only then:`,'display:none']].concat(keys)
    .concat([[`partly pinned (${CR().resolved_dex}-${CR().unresolved_dex} dex)`,
              'background:color-mix(in srgb,var(--text-muted) 35%,transparent);'+
              'border:2px solid var(--text-muted)'],
                      [`not pinned by the data (&gt; ${CR().unresolved_dex} dex)`,
                       'background:transparent;border:2px solid var(--text-muted)'],
                      ['no error','background:var(--text-muted)']]);}
function markerAttrs(st,col,baseW){
  // ONE place deciding how a point is drawn, so the cube and the manifold cannot disagree.
  if(st&&st.ring)return {fill:'none','fill-opacity':.9,stroke:st.stroke,'stroke-width':1.8};
  if(st&&st.halfTone)return {fill:st.fill,'fill-opacity':st.fillOpacity,stroke:st.stroke,'stroke-width':1.5};
  if(st&&st.halo)return {fill:col,'fill-opacity':.9,stroke:'var(--text-primary)','stroke-width':2};
  return {fill:col,'fill-opacity':.9,stroke:'var(--surface-1)','stroke-width':baseW};}
function legendFrom(keys){
  const lg=document.createElement('div');lg.className='legend';
  keys.forEach(([t,s])=>{const sp=document.createElement('span');
    sp.innerHTML=`<i style="${s}"></i>${t}`;lg.appendChild(sp);});
  return lg;}

// --- 1. the quantity table ----------------------------------------------------------------
function quantityTable(idx){
  const box=document.createElement('div');
  const tb=document.createElement('table');
  const head=['quantity','cohorts','claims','decline','coverage@95','±SE','closed cov','(n closed)',
              'pinned by data','|err| dex','unclamped','clamp%','width dex','closed/cap/nocap/none'];
  let html='<tr>'+head.map(h=>`<th>${h}</th>`).join('')+'</tr>';
  const accOnly=D.accuracy_only_quantities||[];
  // A quantity is counted over the cohorts where it EXISTS (external_hazard only where a hazard is
  // planted), and a nuisance quantity gets one row per arm: pooling the hazard arm with realfull
  // would quote one figure for two different questions (sr-customer review, 2026-10-29).
  const rows=[];
  D.quantities.forEach((q,qi)=>{
    const a=D.result.applicable&&D.result.applicable[qi];
    const qIdx=a?idx.filter(i=>a[i]):idx;
    if((D.nuisance_quantities||{})[q])
      [...new Set(qIdx.map(i=>D.cohort.arm[i]))].sort().forEach(arm=>rows.push({q,qi,
        ri:qIdx.filter(i=>D.cohort.arm[i]===arm),
        label:`${q} [arm: ${arm}] <i>(nuisance: would-be claims, not returned by the method)</i>`}));
    else rows.push({q,qi,ri:qIdx,label:q});
  });
  rows.forEach(({q,qi,ri,label})=>{
    const raw=qi<4;
    let hits=[],cl=[],errs=[],unc=[],wid=[],nclamp=0,declined=0,shapes=[0,0,0,0],closedHit=[];
    ri.forEach(i=>{
      const s=D.result.shape[qi][i],h=D.result.hit[qi][i],e=errAt(qi,i);
      shapes[s]++;
      if(s===0)declined++;
      if(h!==null){hits.push(h);cl.push(D.cohort.cluster_key[i]);
        if(s===3)closedHit.push(h);}
      if(e!==null){errs.push(e);
        if(clampedAt(qi,i))nclamp++;else unc.push(e);}
      const w=D.result.width[qi][i]; if(w!==null)wid.push(w);
    });
    const cov=hits.length?hits.reduce((a,b)=>a+b,0)/hits.length:null;
    const se=hits.length?clusterRobustSE(hits,cl):null;
    const ccov=closedHit.length?closedHit.reduce((a,b)=>a+b,0)/closedHit.length:null;
    const f=(v,d)=>v===null?'--':v.toFixed(d===undefined?3:d);
    const isAcc=accOnly.indexOf(q)>=0;
    const tag=(isAcc&&D.result.spread)
      ?` <i>(n=${errs.length}; ${comboMode()==='theta'?'fit-filled, not a claim':'pinned cohorts only'})</i>`:'';
    html+=`<tr${raw?' style="font-weight:600"':''}><td>${label}</td>`+
      `<td>${ri.length}</td>`+
      `<td>${isAcc?'--':hits.length}</td>`+
      `<td>${isAcc?'--':(100*declined/(ri.length||1)).toFixed(0)+'%'}</td>`+
      `<td>${isAcc?'<i>no interval</i>':f(cov)}</td>`+
      `<td>${isAcc?'--':(se===null?'--':'±'+se.toFixed(3))}</td>`+
      `<td>${isAcc?'--':f(ccov)}</td>`+
      `<td>${isAcc?'--':closedHit.length}</td>`+
      `<td>${(isAcc&&D.result.spread)?(()=>{const k=comboCounts(qi,ri);
         return k.resolved+'/'+(k.n-k.unknown);})():'--'}</td>`+
      `<td>${f(median(errs))}${tag}</td>`+
      `<td>${f(median(unc))}</td>`+
      `<td>${errs.length?(100*nclamp/errs.length).toFixed(0)+'%':'--'}</td>`+
      `<td>${isAcc?'--':f(median(wid))}</td>`+
      `<td>${isAcc?'--':shapes[3]+'/'+shapes[1]+'/'+shapes[2]+'/'+shapes[0]}</td></tr>`;
  });
  tb.innerHTML=html;
  const wrap=document.createElement('div');wrap.className='scrollx';wrap.appendChild(tb);
  box.appendChild(wrap);
  box.appendChild(denomLine(
    'claims is how many cohorts the method made any statement about; non-claims are excluded '+
    'from the coverage denominator, so coverage and claims must be read together -- declining '+
    'more often raises coverage for free. closed cov is restricted to two-sided intervals, the '+
    'only column comparable with width. |err| is floored; unclamped is the same median where the '+
    'floor did no work, and the two diverge exactly where the floor is doing it. ±SE is '+
    'cluster-robust, clustered on the parameter point. A combination has a point estimate but no '+
    'interval, so its coverage columns are blank rather than invented. pinned by data counts the '+
    'cohorts whose own near-optimal fits agree on the combination to within '+CR().resolved_dex+
    ' dex; under the resolvability toggle a combination\'s |err| is the median over those only.'));
  return box;}

// --- 2/3. the two cubes, sharing one camera -----------------------------------------------
const CAM={rx:0.45,ry:0.7,zoom:1};
const CUBE_REDRAW=[];
function cube(idx,qi,mode){
  const W=430,H=300;
  const box=document.createElement('div');
  const pts=[];
  byPoint(idx).forEach((ii,k)=>{
    const x=D.cohort.rho_beta[ii[0]],y=D.cohort.rho_eta[ii[0]],z=D.cohort.rho_eps[ii[0]];
    if(!(x>0&&y>0&&z>0))return;
    let v=null;
    let sty=null;
    if(mode==='acc'){sty=pointStyle(qi,ii);v=sty.v;}
    else{const h=ii.map(i=>D.result.hit[qi][i]).filter(t=>t!==null);
         v=h.length?h.reduce((a,b)=>a+b,0)/h.length:null;}
    pts.push({x:Math.log10(x),y:Math.log10(y),z:Math.log10(z),v,n:ii.length,
              sty,k:D.cohort.kappa[ii[0]],claims:mode==='acc'?ii.length:
              ii.map(i=>D.result.hit[qi][i]).filter(t=>t!==null).length});
  });
  if(!pts.length){box.appendChild(denomLine(
    'no cohort in this selection has a log-position (a parameter on a box boundary has none)'));
    return box;}
  const ext=a=>{const v=pts.map(p=>p[a]);return [Math.min(...v),Math.max(...v)];};
  const [x0,x1]=ext('x'),[y0,y1]=ext('y'),[z0,z1]=ext('z');
  const nrm=(v,a,b)=>((v-a)/((b-a)||1))-0.5;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  box.appendChild(svg);
  const draw=()=>{
    while(svg.firstChild)svg.removeChild(svg.firstChild);
    const cr=Math.cos(CAM.rx),sr=Math.sin(CAM.rx),cy=Math.cos(CAM.ry),sy2=Math.sin(CAM.ry);
    const S=95*CAM.zoom, CX=W/2, CY=H/2+6;
    const proj=(a,b,c)=>{const X=a*cy-c*sy2, Z=a*sy2+c*cy, Y=b*cr-Z*sr, D2=b*sr+Z*cr;
      return [CX+X*S, CY-Y*S, D2];};
    // Axis tripod, so the rotation is readable as an orientation rather than a jumble.
    const O=proj(-0.5,-0.5,-0.5);
    [['rho_beta',[0.5,-0.5,-0.5]],['rho_eta',[-0.5,0.5,-0.5]],['rho_eps',[-0.5,-0.5,0.5]]]
      .forEach(([lab,e])=>{const P2=proj(e[0],e[1],e[2]);
        svg.appendChild(el('line',{x1:O[0],y1:O[1],x2:P2[0],y2:P2[1],class:'axis'}));
        svg.appendChild(el('text',{x:P2[0],y:P2[1]-3,'text-anchor':'middle',class:'ticklab'},lab));});
    const proj2=pts.map(p=>{const q2=proj(nrm(p.x,x0,x1),nrm(p.y,y0,y1),nrm(p.z,z0,z1));
      return {p,X:q2[0],Y:q2[1],Z:q2[2]};});
    proj2.sort((a,b)=>a.Z-b.Z);                       // painter's algorithm
    proj2.forEach(o=>{
      const st=mode==='acc'?o.p.sty:null;
      const col=mode==='acc'?st.fill:covColour(o.p.v);
      const c=el('circle',Object.assign({cx:o.X,cy:o.Y,r:3.2+1.6*(CAM.zoom-1)},
        markerAttrs(st,col,1)));
      hoverable(c,`kappa ${o.p.k} · log10 rho_beta ${o.p.x.toFixed(2)} · rho_eta `+
        `${o.p.y.toFixed(2)} · rho_eps ${o.p.z.toFixed(2)} · `+
        (mode==='acc'?`median |err| ${o.p.v===null?'--':o.p.v.toFixed(3)} dex${st.extra}`
                     :(o.p.v===null?'no claim':`coverage ${o.p.v.toFixed(3)} (${o.p.claims} claims)`)));
      svg.appendChild(c);});
  };
  draw();CUBE_REDRAW.push(draw);
  let drag=null;
  svg.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];svg.setPointerCapture(e.pointerId);});
  svg.addEventListener('pointerup',()=>{drag=null;});
  svg.addEventListener('pointermove',e=>{if(!drag)return;
    CAM.ry+=(e.clientX-drag[0])*0.01; CAM.rx+=(e.clientY-drag[1])*0.01;
    CAM.rx=Math.max(-1.5,Math.min(1.5,CAM.rx)); drag=[e.clientX,e.clientY];
    CUBE_REDRAW.forEach(f=>f());});
  svg.addEventListener('wheel',e=>{e.preventDefault();
    CAM.zoom=Math.max(0.4,Math.min(3,CAM.zoom*(e.deltaY<0?1.1:0.9)));
    CUBE_REDRAW.forEach(f=>f());},{passive:false});
  box.appendChild(legendFrom(mode==='acc'?accLegendKeys(qi)
    : [['&ge;0.90','background:var(--s6)'],['&ge;0.80','background:var(--s4)'],
       ['below','background:var(--s2)'],['never claimed','background:var(--text-muted)']]));
  if(mode==='acc'&&comboNote(qi,idx))box.appendChild(denomLine(comboNote(qi,idx)));
  box.appendChild(denomLine(`${pts.length} parameter points (not cohorts -- seed replicates are `+
    `collapsed, so a thin design cannot look dense). Drag to rotate, scroll to zoom; both cubes `+
    `share one camera.`));
  return box;}

// --- 4/5. dual-axis trends ----------------------------------------------------------------
function dualAxis(rows,xlab,xfmt,note){
  // Left axis coverage 0..1 with the nominal line; right axis median |err| in dex, auto-scaled.
  const W=440,H=236,P=48,R=42;
  const box=document.createElement('div');
  const ok=rows.filter(r=>r.cov!==null||r.err!==null);
  if(ok.length<2){box.appendChild(denomLine('not enough bins to read a trend'));return box;}
  const xs=ok.map(r=>r.x);
  let x0=Math.min(...xs),x1=Math.max(...xs); if(x1<=x0){x0-=0.5;x1+=0.5;}
  const es=ok.map(r=>r.err).filter(v=>v!==null&&isFinite(v));
  const e1=es.length?Math.max(...es):1;
  const sx=v=>P+(W-P-R)*((v-x0)/((x1-x0)||1));
  const sy=v=>H-P-(H-P-18)*v;                       // coverage 0..1
  const sye=v=>H-P-(H-P-18)*(v/(e1||1));            // error, right scale
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  axes(svg,{x0,x1,y0:0,y1:1,sx,sy,W,H,P,xlab,ylab:'coverage',yticks:4,
            xfmt:xfmt||fmtNum,yfmt:v=>v.toFixed(2)});
  const lv=D.provenance.level;
  if(lv!==null&&lv!==undefined){
    svg.appendChild(el('line',{x1:P,y1:sy(lv),x2:W-R,y2:sy(lv),class:'nominal'}));
    svg.appendChild(el('text',{x:W-R-2,y:sy(lv)-3,'text-anchor':'end',class:'ticklab'},
                      `nominal ${lv}`));}
  niceTicks(0,e1,4).forEach(t=>{const y=sye(t);if(y<8||y>H-P)return;
    svg.appendChild(el('text',{x:W-R+6,y:y+3,class:'ticklab'},fmtNum(t)));});
  svg.appendChild(el('text',{x:W-6,y:14,'text-anchor':'end',class:'axlab'},'median |err| dex'));
  let dc='',de='';
  ok.forEach(r=>{if(r.cov!==null)dc+=(dc?'L':'M')+sx(r.x)+' '+sy(r.cov)+' ';});
  ok.forEach(r=>{if(r.err!==null)de+=(de?'L':'M')+sx(r.x)+' '+sye(r.err)+' ';});
  if(dc)svg.appendChild(el('path',{d:dc,fill:'none',stroke:'var(--s1)','stroke-width':2}));
  if(de)svg.appendChild(el('path',{d:de,fill:'none',stroke:'var(--s4)','stroke-width':2,
                                   'stroke-dasharray':'5 3'}));
  ok.forEach(r=>{
    if(r.cov!==null){const c=el('circle',{cx:sx(r.x),cy:sy(r.cov),r:4,fill:'var(--s1)',
      stroke:'var(--surface-1)','stroke-width':2});
      hoverable(c,`${r.label} · coverage ${r.cov.toFixed(3)} (${r.claims} claims`+
        (r.points?`, ${r.points} points`:'')+')');svg.appendChild(c);}
    if(r.err!==null){const c=el('circle',{cx:sx(r.x),cy:sye(r.err),r:3.4,fill:'var(--s4)',
      stroke:'var(--surface-1)','stroke-width':2});
      hoverable(c,`${r.label} · unclamped |err| ${r.err.toFixed(3)} dex (${r.nerr})`);
      svg.appendChild(c);}});
  box.appendChild(svg);
  const lg=document.createElement('div');lg.className='legend';
  lg.innerHTML='<span><i style="background:var(--s1)"></i>coverage (left)</span>'+
               '<span><i style="background:var(--s4)"></i>median |err| dex (right)</span>';
  box.appendChild(lg);
  if(note)box.appendChild(denomLine(note));
  return box;}

// --- 6. the manifold ----------------------------------------------------------------------
function manifoldScatter(idx,qi){
  const W=440,H=280,P=50;
  const box=document.createElement('div');
  const pts=[];
  byPoint(idx).forEach(ii=>{
    const lr=D.cohort.log_ratio[ii[0]], cv=median(ii.map(i=>D.cohort.cv[i]));
    if(lr===null||cv===null||!isFinite(lr)||!isFinite(cv))return;
    const sty=pointStyle(qi,ii);
    pts.push({x:lr,y:cv,v:sty.v,sty,n:ii.length});});
  if(!pts.length){box.appendChild(denomLine(
    'no cohort in this selection carries both a removal ratio and a measurable CV'));return box;}
  let x0=Math.min(...pts.map(p=>p.x)),x1=Math.max(...pts.map(p=>p.x));
  if(x1<=x0){x0-=0.5;x1+=0.5;}
  const y1=Math.max(0.05,Math.max(...pts.map(p=>p.y)));
  const sx=v=>P+(W-P-10)*((v-x0)/((x1-x0)||1));
  const sy=v=>H-P-(H-P-16)*(v/y1);
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  axes(svg,{x0,x1,y0:0,y1,sx,sy,W,H,P,xlab:'log10 removal ratio  rho_beta/rho_eta',
            ylab:'CV of observed deaths',yfmt:v=>v.toFixed(2)});
  pts.forEach(p=>{const st=p.sty;
    const c=el('circle',Object.assign({cx:sx(p.x),cy:sy(p.y),r:4.2},markerAttrs(st,st.fill,1.2)));
    hoverable(c,`ratio 10^${p.x.toFixed(2)} · CV ${p.y.toFixed(3)} · `+
      `median |err| ${p.v===null?'--':p.v.toFixed(3)} dex · ${p.n} cohorts${st.extra}`);
    svg.appendChild(c);});
  box.appendChild(svg);
  box.appendChild(legendFrom(accLegendKeys(qi)));
  if(comboNote(qi,idx))box.appendChild(denomLine(comboNote(qi,idx)));
  box.appendChild(denomLine(`${pts.length} parameter points. CV is MEASURED from each cohort's `+
    `own observed deaths -- it is not stored on the spec, only its band survives in the stratum.`));
  return box;}

function lineWithN(rows,xlab,ylab,q){
  const W=440,H=236,P=48;
  const box=document.createElement('div');
  const ok=rows.filter(r=>r.rate!==null);
  if(!ok.length){box.appendChild(denomLine('nothing to show'));return box;}
  const xs=ok.map(r=>Math.log10(r.k));
  let x0=Math.min(...xs),x1=Math.max(...xs);
  if(x1<=x0){x0-=0.5;x1+=0.5;}
  const sx=v=>P+(W-P-12)*((v-x0)/((x1-x0)||1));
  const sy=v=>H-P-(H-P-18)*v;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  // A rate axis with no numbers on it is the same complaint as the other panels: 0..1 is not
  // obvious, and neither is where 0.5 sits. x is log10 kappa, labelled as kappa itself.
  axes(svg,{x0,x1,y0:0,y1:1,sx,sy,W,H,P,xlab,ylab,yticks:4,
            xfmt:t=>fmtNum(Math.pow(10,t)),yfmt:v=>v.toFixed(2)});
  let d='';ok.forEach((r,i)=>{d+=(i?'L':'M')+sx(Math.log10(r.k))+' '+sy(r.rate)+' ';});
  svg.appendChild(el('path',{d:d,fill:'none',stroke:'var(--s1)','stroke-width':2}));
  ok.forEach(r=>{const c=el('circle',{cx:sx(Math.log10(r.k)),cy:sy(r.rate),r:4,
    fill:'var(--s1)',stroke:'var(--surface-1)','stroke-width':2});
    hoverable(c,`kappa=${r.k} · decline ${r.rate.toFixed(3)} · n=${r.n}`);svg.appendChild(c);});
  box.appendChild(svg);
  const tb=document.createElement('table');
  tb.innerHTML='<tr><th>kappa</th><th>decline</th><th>n</th></tr>'+
    ok.map(r=>`<tr><td>${r.k}</td><td>${r.rate.toFixed(3)}</td><td>${r.n}</td></tr>`).join('');
  box.appendChild(tb);
  return box;}
"""

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%TITLE%</title><style>%CSS%</style></head>
<body class="viz-root">
<h1>%TITLE%</h1>
<p class="sub">%SUBTITLE%</p>
<div class="prov">%PROVENANCE%</div>
%ARMSPANEL%
<p class="filternote on" id="filternote"></p>
<div class="ctl">
  <span><label for="q">quantity</label><select id="q"></select></span>
  <span><label for="coh">cohorts</label><select id="coh"></select></span>
  <span><label for="val">data</label><select id="val"></select></span>
  <span><label for="cmode">combinations</label><select id="cmode">
    <option value="resolve">resolvability: is it pinned by the data?</option>
    <option value="theta">error at the fitted point (flagged)</option></select></span>
  <span><label for="theme">theme</label>
    <select id="theme"><option value="">system</option><option value="light">light</option>
    <option value="dark">dark</option></select></span>
</div>
<div class="views" id="views"></div>
<script>var DATA=%DATA%;</script>
<script>%HELPERS%
const D=DATA;const VIEWS=%VIEWS%;
const ui={
  quantity:()=>document.getElementById('q').value,
  cohortFilter:()=>{const e=document.getElementById('coh');return e?e.value:'well';},
  // THE COHORT SELECTION every view reads. Default `well`: only cohorts whose generating model
  // the method actually fits. A cohort carrying a feature the method does not model measures
  // MISSPECIFICATION -- worth having, and the reason the feature arms exist -- but pooling it with
  // a calibration result silently answers a different question than the reader thinks they asked.
  dataFilter:()=>{const e=document.getElementById('val');return e?e.value:'valid';},
  // VALIDITY IS A SEPARATE QUESTION FROM SPECIFICATION, so it is a separate control and the two
  // are ANDed. `well_specified` asks whether the METHOD fits the model that generated the cohort;
  // validity asks whether the COHORT'S DATA is trustworthy at all. A figure needs both to mean
  // what a reader will take it to mean.
  keepByValidity:(i)=>{
    const f=ui.dataFilter(); if(f==='any')return true;
    const v=(D.cohort.valid||[])[i];
    if(v===undefined||v===null)return f==='unaudited';
    return f==='valid'? v===1 : f==='invalid'? v===0 : true;},
  idx:()=>{
    const f=ui.cohortFilter(), out=[];
    for(let i=0;i<D.n_cohorts;i++){
      if(!ui.keepByValidity(i))continue;
      if(f==='all'){out.push(i);continue;}
      if(f==='well'){if(D.cohort.well_specified&&D.cohort.well_specified[i])out.push(i);continue;}
      if(f==='mis'){if(D.cohort.well_specified&&!D.cohort.well_specified[i])out.push(i);continue;}
      if(f.slice(0,5)==='with:'){
        if((D.cohort.features&&D.cohort.features[i]||[]).indexOf(f.slice(5))>=0)out.push(i);
        continue;}
      if(f.slice(0,8)==='without:'){
        if((D.cohort.features&&D.cohort.features[i]||[]).indexOf(f.slice(8))<0)out.push(i);
        continue;}
      if(f.slice(0,4)==='arm:'){if(D.cohort.arm[i]===f.slice(4))out.push(i);continue;}
      if(f.slice(0,6)==='group:'){if((D.cohort.group||[])[i]===f.slice(6))out.push(i);continue;}
      out.push(i);
    }
    return out;},
  // Cohorts where the SELECTED quantity exists at all. external_hazard applies only to cohorts
  // carrying a hazard, so without this a pure cohort reads as a decline of something it never had.
  applies:(qi)=>{const a=D.result.applicable;
    if(!a||!a[qi])return ui.idx();
    return ui.idx().filter(i=>a[qi][i]);},
  filterLabel:()=>{const e=document.getElementById('coh');
    return e&&e.selectedIndex>=0?e.options[e.selectedIndex].textContent:'';},
};
function render(){
  const host=document.getElementById('views');host.innerHTML='';
  const banner=document.getElementById('filternote');
  if(banner){const n=ui.idx().length;
    const vsrc=D.provenance.validity_source;
    const vlab=ui.dataFilter()==='any'?'any validity':ui.dataFilter()+' data only';
    banner.textContent=`Showing ${n} of ${D.n_cohorts} cohorts — ${ui.filterLabel()}, ${vlab}`+
      (vsrc&&ui.dataFilter()!=='any'?` (audit: ${vsrc})`:'')+'. '+
      [ui.cohortFilter()==='well'
         ? 'Only cohorts whose generating model this method actually fits -- the rest measure '+
           'misspecification.'
         : null,
       ui.dataFilter()==='valid'
         ? 'Only cohorts the audit found sound -- the rest carry a known generator defect '+
           '(inflated lifespan CV, or a death with zero probability mass), so a figure including '+
           'them is partly measuring that defect.'
         : null,
       'Every figure below is drawn from this selection.'
      ].filter(Boolean).join(' ');
    banner.className=ui.cohortFilter()==='all'?'filternote':'filternote on';}
  const nq=(D.nuisance_quantities||{})[ui.quantity()];
  if(nq){const w=document.createElement('div');w.className='warn';w.style.gridColumn='1 / -1';
    w.textContent=ui.quantity()+' is a NUISANCE here. Every figure for it is a WOULD-BE claim of the '+
      'method, read from '+nq.source+' and NOT returned by the method; it enters no verdict. The '+
      'method has no heterogeneity term, so heterogeneous cohorts are misspecified for it. Read '+
      'the arms apart: the table gives one row per arm.';
    host.appendChild(w);}
  for(const v of VIEWS){
    const card=document.createElement('div');card.className='card';
    card.appendChild(Object.assign(document.createElement('h2'),{textContent:v.title}));
    card.appendChild(Object.assign(document.createElement('p'),
      {className:'desc',textContent:v.description}));
    try{card.appendChild(v.render());}
    catch(e){card.appendChild(denomLine('view failed: '+e.message));}
    host.appendChild(card);
  }
}
document.addEventListener('DOMContentLoaded',()=>{
  tipInit();
  const q=document.getElementById('q');
  D.quantities.forEach(n=>q.appendChild(Object.assign(document.createElement('option'),
    {value:n,textContent:n})));
  q.value=D.quantities[0];q.addEventListener('change',render);
  const coh=document.getElementById('coh');
  const nWell=(D.cohort.well_specified||[]).reduce((t,v)=>t+(v||0),0);
  const opts=[['well',`correctly specified only (${nWell})`],
              ['all',`all cohorts (${D.n_cohorts})`]];
  if(nWell<D.n_cohorts)opts.push(['mis',`misspecified only (${D.n_cohorts-nWell})`]);
  const feats=[...new Set((D.cohort.features||[]).flat())].sort();
  feats.forEach(f=>{
    const n=(D.cohort.features||[]).filter(x=>x.indexOf(f)>=0).length;
    opts.push(['with:'+f,`with ${f} (${n})`]);
    opts.push(['without:'+f,`without ${f} (${D.n_cohorts-n})`]);});
  [...new Set(D.cohort.arm)].sort().forEach(a=>{
    const n=D.cohort.arm.filter(x=>x===a).length;opts.push(['arm:'+a,`arm: ${a} (${n})`]);});
  const groups=[...new Set(D.cohort.group||[])].sort();
  if(groups.length>1)groups.forEach(g=>{
    const n=(D.cohort.group||[]).filter(x=>x===g).length;
    opts.push(['group:'+g,`group: ${g} (${n})`]);});
  opts.forEach(([v,t])=>coh.appendChild(Object.assign(document.createElement('option'),
    {value:v,textContent:t})));
  coh.value='well';coh.addEventListener('change',render);

  const val=document.getElementById('val');
  const vs=D.cohort.valid||[];
  const nV=vs.filter(x=>x===1).length, nI=vs.filter(x=>x===0).length,
        nU=vs.filter(x=>x===undefined||x===null).length;
  const vopts=[];
  if(nV||nI){
    vopts.push(['valid',`valid only (${nV})`]);
    vopts.push(['any',`any validity (${D.n_cohorts})`]);
    if(nI)vopts.push(['invalid',`invalid only (${nI})`]);
  } else {
    vopts.push(['any','no validity audit loaded']);
  }
  if(nU)vopts.push(['unaudited',`not audited (${nU})`]);
  vopts.forEach(([v,t])=>val.appendChild(Object.assign(document.createElement('option'),
    {value:v,textContent:t})));
  val.value=(nV||nI)?'valid':'any';
  val.addEventListener('change',render);
  document.getElementById('cmode').addEventListener('change',render);
  document.getElementById('theme').addEventListener('change',e=>{
    if(e.target.value)document.documentElement.setAttribute('data-theme',e.target.value);
    else document.documentElement.removeAttribute('data-theme');});
  render();
});
</script>
</body></html>
"""


#: How each state reads to someone who has not been told what an "arm" is.
_STATE_LABEL = {
    "none": "n/a",
    "handled": "modelled",
    "invisible": "NOT modelled",
    "undeclared": "not declared",
    "unknown": "unknown",
    "mixed": "see above",
}


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if s is not None else "")


def _arms_panel(payload: dict) -> str:
    """The collapsed "what is in each cohort group, and was it fitted?" block.

    Generated from the payload -- the method's declared `capabilities` against the feature each arm
    plants -- rather than written out, so it cannot drift from what the method actually claims.
    """
    arms = payload.get("arms") or []
    if not arms:
        return ""
    prov = payload.get("provenance") or {}
    caps = prov.get("method_capabilities")
    rows = []
    for r in arms:
        st = r.get("state", "unknown")
        rows.append(
            f"<tr><td><b>{_esc(r['arm'])}</b></td><td>{r.get('n', 0)}</td>"
            f"<td>{_esc(r.get('feature') or '--')}</td>"
            f"<td class='state state-{_esc(st)}'>{_esc(_STATE_LABEL.get(st, st))}</td>"
            f"<td>{_esc(r.get('detail'))}</td></tr>")
    declared = (", ".join(caps) if caps else "none")
    if caps is None:
        declared = "unknown (the method could not be loaded here)"
    return (
        "<details class='arms'><summary>What is in each cohort group &mdash; and what was "
        "actually fitted</summary>"
        "<p>Every cohort is fitted for the same four SR parameters: "
        "<b>eta, beta, epsilon, kappa</b>. The groups differ in what else is present in the data. "
        f"This method declares it models: <b>{_esc(declared)}</b>.</p>"
        "<p><b>No method is ever handed a feature's true value.</b> An <code>Observation</code> "
        "carries only the ages, the event flags, the recording resolution and the fit window &mdash; "
        "it contains no truth. So where a feature is invisible, the cohort is fitted <b>under the "
        "assumption that the feature is absent</b>, and the result measures misspecification "
        "rather than calibration.</p>"
        "<table><tr><th>group</th><th>cohorts</th><th>what the data carries</th>"
        "<th>modelled?</th><th></th></tr>" + "".join(rows) + "</table>"
        "<p><i>not declared</i> means the observation does carry the feature, but the method does "
        "not declare modelling it &mdash; so whether it is handled cannot be read off the contract, "
        "and the cohort is left OUT of the default correctly-specified view.</p>"
        "<p>Right-censoring is the case worth knowing about. Both shipped methods compute the "
        "censored term correctly (log-survival at the recorded age), yet declaring it would be "
        "wrong <i>for this library</i>: a cohort is censored at a sample quantile and its ages are "
        "floored to the recording resolution only afterwards, so deaths land tied to the censoring "
        "age and are scored over a bin they did not die in. 287 of 304 censor-bearing cohorts carry "
        "such ties. &ldquo;The term is right&rdquo; and &ldquo;these cohorts are well specified&rdquo; "
        "are different claims, and only the first holds.</p></details>")


def assemble(payload: dict, out_dir: str, title: str = "SR calibration method explorer") -> str:
    """Write the self-contained explorer. Raises if any view wants a field the payload lacks."""
    views = all_views()
    check_requirements(payload, views)

    prov = payload["provenance"]
    ref = (f"<b>reference</b> {prov.get('reference_method') or '(none set)'}"
           f" v{prov.get('reference_version') or '?'}"
           f" (run {prov.get('reference_run_id') or '?'})")
    absent = prov.get("arms_absent") or []
    prov_html = (
        f"<b>method</b> {prov['method_name']} v{prov['method_version']} &nbsp;·&nbsp; {ref}<br>"
        f"<b>library</b> {prov['library_version']} &nbsp;·&nbsp; "
        f"<b>contract</b> {prov['contract_version']} &nbsp;·&nbsp; "
        f"<b>cohorts</b> {payload['n_cohorts']} &nbsp;·&nbsp; "
        f"generated {prov['generated_at']}<br>"
        f"<b>arms present</b> {', '.join(prov['arms_present'])}"
    )
    if absent:
        prov_html += (f"<div class='warn'><b>UNTESTED REGIMES in this run:</b> "
                      f"{', '.join(absent)}. No figure below says anything about them.</div>")

    arms_html = _arms_panel(payload)

    views_js = "[" + ",".join(
        "{{title:{t},description:{d},render:function(){{{body}}}}}".format(
            t=json.dumps(v.title), d=json.dumps(v.description), body=v.render_js)
        for v in views) + "]"

    helpers = _JS_HELPERS.replace("%ARMS%", json.dumps(ARM_ORDER))
    html = (_PAGE
            .replace("%TITLE%", title)
            .replace("%SUBTITLE%", "Every figure carries its denominator. Arms are never pooled. "
                                   "Declines are reported beside coverage, never folded into it.")
            .replace("%CSS%", _CSS)
            .replace("%PROVENANCE%", prov_html)
            .replace("%ARMSPANEL%", arms_html)
            .replace("%DATA%", json.dumps(payload, separators=(",", ":")))
            .replace("%HELPERS%", helpers)
            .replace("%VIEWS%", views_js))

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "explorer.html")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(html)
    return p
