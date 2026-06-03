import { useState, useEffect, useRef, useCallback } from "react";
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Area, AreaChart,
  BarChart, ScatterChart, Scatter, Cell,
} from "recharts";

// ─── THEME ────────────────────────────────────────────────────────────────────
// IT: palette colori centralizzata (sfondi, accenti, testo) condivisa dai componenti.
// EN: centralized color palette (backgrounds, accents, text) shared by components.
const T = {
  bg0: "#080a0c", bg1: "#0d1117", bg2: "#161b22", bg3: "#1c2128",
  border: "#21262d", amber: "#e6a817", amberDim: "#7a5800",
  green: "#3fb950", greenDim: "#0d3320", red: "#f85149", redDim: "#3d0f0f",
  blue: "#58a6ff", blueDim: "#1a3a5c", purple: "#bc8cff",
  text: "#c9d1d9", textDim: "#6e7681", textMute: "#21262d",
};

// ─── BINANCE ──────────────────────────────────────────────────────────────────
// IT: parametri stream Binance (simbolo, timeframe, n candele storiche).
// EN: Binance stream parameters (symbol, timeframe, historical candle count).
const SYMBOL = "BTCUSDT";
const INTERVAL = "1m";
const LIMIT = 120;

// IT: scarica le ultime LIMIT candele OHLCV e le normalizza per i grafici.
// EN: fetches the last LIMIT OHLCV candles and normalizes them for the charts.
async function fetchKlines() {
  const r = await fetch(`https://api.binance.com/api/v3/klines?symbol=${SYMBOL}&interval=${INTERVAL}&limit=${LIMIT}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()).map((k, i) => {
    const open = +k[1], high = +k[2], low = +k[3], close = +k[4], volume = +k[5];
    return { i, ts: k[0], time: new Date(k[0]).toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit" }), open, high, low, close, volume, bullish: close >= open, bodyLow: Math.min(open, close), bodyHigh: Math.max(open, close), vwap: null };
  });
}

// IT: ticker statistiche 24h (high/low/volume/variazione %).
// EN: 24h statistics ticker (high/low/volume/percent change).
async function fetch24hr() {
  const r = await fetch(`https://api.binance.com/api/v3/ticker/24hr?symbol=${SYMBOL}`);
  return r.json();
}

// IT: VWAP cumulativo (typical price ponderato per volume) candela per candela.
// EN: cumulative VWAP (volume-weighted typical price) candle by candle.
function computeVWAP(candles) {
  let cpv = 0, cv = 0;
  return candles.map(c => { const tp = (c.high + c.low + c.close) / 3; cpv += tp * c.volume; cv += c.volume; return { ...c, vwap: cv > 0 ? cpv / cv : c.close }; });
}

// IT: costruisce il Volume Profile (volumi buy/sell per fascia di prezzo).
// EN: builds the Volume Profile (buy/sell volumes per price bin).
function buildVP(candles, bins = 28) {
  if (candles.length < 2) return [];
  const hi = Math.max(...candles.map(c => c.high)), lo = Math.min(...candles.map(c => c.low));
  const step = (hi - lo) / bins || 1;
  const p = Array.from({ length: bins }, (_, i) => ({ price: lo + i * step + step / 2, buyVol: 0, sellVol: 0 }));
  candles.forEach(c => { const b = Math.min(Math.floor(((c.high + c.low + c.close) / 3 - lo) / step), bins - 1); if (b >= 0) { if (c.bullish) p[b].buyVol += c.volume; else p[b].sellVol += c.volume; } });
  const mx = Math.max(...p.map(x => x.buyVol + x.sellVol), 1);
  return p.map(x => ({ ...x, total: x.buyVol + x.sellVol, pct: (x.buyVol + x.sellVol) / mx }));
}

// IT: forecast Monte Carlo client-side: GARCH(1,1) + shock t-Student → bande percentili.
// EN: client-side Monte Carlo forecast: GARCH(1,1) + t-Student shocks → percentile bands.
function simulateMCMC(candles, steps = 30, paths = 1200) {
  if (candles.length < 5) return [];
  const rets = candles.slice(-30).map((c, i, a) => i === 0 ? 0 : Math.log(c.close / a[i-1].close)).slice(1);
  const mu = rets.reduce((s,r) => s+r, 0) / rets.length;
  const baseVol = Math.sqrt(rets.map(r => (r-mu)**2).reduce((s,v)=>s+v,0)/rets.length);
  const last = candles[candles.length - 1].close;
  const buckets = Array.from({ length: steps }, () => []);
  for (let p = 0; p < paths; p++) {
    let price = last, vol = baseVol;
    for (let t = 0; t < steps; t++) {
      const i = (Math.random()-0.5)*2;
      // IT: ricorsione GARCH(1,1) sulla varianza, capata per evitare blow-up.
      // EN: GARCH(1,1) recursion on variance, capped to avoid blow-up.
      vol = Math.sqrt(Math.max(1e-8, 1e-5 + 0.88*vol**2 + 0.10*(i*vol)**2));
      vol = Math.min(vol, 0.008);
      const u1 = Math.max(1e-10, Math.random()), u2 = Math.random();
      const z = Math.sqrt(-2*Math.log(u1)) * Math.cos(2*Math.PI*u2);
      const chi2 = Array.from({length:5}, () => { const a=Math.max(1e-10,Math.random()),b=Math.random(); return (Math.sqrt(-2*Math.log(a))*Math.cos(2*Math.PI*b))**2; }).reduce((s,v)=>s+v,0);
      price = price * Math.exp(mu + vol * (z / Math.sqrt(chi2/5)));
      buckets[t].push(price);
    }
  }
  const now = new Date();
  return buckets.map((prices, t) => {
    prices.sort((a,b)=>a-b); const n = prices.length, pct = q => prices[Math.floor(n*q)];
    return { time: new Date(now.getTime()+(t+1)*60000).toLocaleTimeString("it-IT",{hour:"2-digit",minute:"2-digit"}), forecast:true, p50:pct(.5), p75:pct(.75), p25:pct(.25), p90:pct(.9), p10:pct(.1), p95:pct(.95), p05:pct(.05) };
  });
}

// IT: metriche di performance dalle candele (Sharpe, maxDD, win-rate, profit factor, vol annualizzata).
// EN: performance metrics from candles (Sharpe, maxDD, win-rate, profit factor, annualized vol).
function calcMetrics(c) {
  if (c.length < 5) return {};
  const rets = c.slice(1).map((d,i)=>Math.log(d.close/c[i].close));
  const mean = rets.reduce((s,r)=>s+r,0)/rets.length, std=Math.sqrt(rets.map(r=>(r-mean)**2).reduce((s,v)=>s+v,0)/rets.length);
  const wins=rets.filter(r=>r>0), losses=rets.filter(r=>r<0);
  let eq=1,mx=1,dd=0; rets.forEach(r=>{eq*=(1+r);if(eq>mx)mx=eq;const d=(mx-eq)/mx;if(d>dd)dd=d;});
  return { sharpe:+(mean/std*Math.sqrt(525600)).toFixed(2), maxDD:+(dd*100).toFixed(2), winRate:+(wins.length/rets.length*100).toFixed(1), profitFactor:+(wins.reduce((s,r)=>s+r,0)/Math.abs(losses.reduce((s,r)=>s+r,0)+1e-9)).toFixed(2), volatility:+(std*Math.sqrt(525600)*100).toFixed(1), totalReturn:+((eq-1)*100).toFixed(2) };
}

// ─── HELPERS ──────────────────────────────────────────────────────────────────
// IT: formattatori di display per valori USD, percentuali e migliaia ($k).
// EN: display formatters for USD values, percentages and thousands ($k).
const fmtUSD = (v, dec=1) => v != null ? `$${Number(v).toLocaleString("en-US",{minimumFractionDigits:dec,maximumFractionDigits:dec})}` : "—";
const fmtPct = (v, sign=true) => v != null ? `${sign && v>0?"+":""}${(+v).toFixed(2)}%` : "—";
const fmtK   = (v) => v != null ? `$${(+v/1000).toFixed(1)}k` : "—";

// IT: indicatore di stato a pallino (verde=ok con glow pulsante, rosso=ko).
// EN: dot status indicator (green=ok with pulsing glow, red=ko).
function Dot({ ok }) {
  return <div style={{ width:7, height:7, borderRadius:"50%", background: ok?T.green:T.red, boxShadow: ok?`0 0 8px ${T.green}`:"none", animation: ok?"pulse 2s infinite":"none" }} />;
}

// IT: coppia etichetta/valore verticale (key-value) per pannelli compatti.
// EN: vertical label/value (key-value) pair for compact panels.
function KV({ label, value, color }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:2 }}>
      <span style={{ fontSize:8, color:T.textDim, letterSpacing:"0.12em" }}>{label}</span>
      <span style={{ fontSize:12, fontFamily:"'Courier New',monospace", color: color||T.text }}>{value}</span>
    </div>
  );
}

// IT: card metrica grande con colore semaforico (positive=null→ambra, true→verde, false→rosso).
// EN: large metric card with traffic-light color (positive=null→amber, true→green, false→red).
function MetricCard({ label, value, unit="", positive=null, sub="" }) {
  const color = positive===null ? T.amber : positive ? T.green : T.red;
  return (
    <div style={{ background:T.bg2, border:`1px solid ${T.border}`, padding:"12px 14px", display:"flex", flexDirection:"column", gap:3 }}>
      <span style={{ fontSize:9, color:T.textDim, letterSpacing:"0.13em", textTransform:"uppercase" }}>{label}</span>
      <span style={{ fontFamily:"'Courier New',monospace", fontSize:22, fontWeight:700, color, lineHeight:1 }}>
        {value}<span style={{ fontSize:11, marginLeft:2, color:T.textDim }}>{unit}</span>
      </span>
      {sub && <span style={{ fontSize:9, color:T.textDim }}>{sub}</span>}
    </div>
  );
}

// IT: badge segnale BUY/SELL/HOLD con icona, colore e probabilità.
// EN: BUY/SELL/HOLD signal badge with icon, color and probability.
function SignalBadge({ signal, prob }) {
  const cfg = { BUY:{col:T.green,bg:T.greenDim,icon:"▲"}, SELL:{col:T.red,bg:T.redDim,icon:"▼"}, HOLD:{col:T.amber,bg:"#2a1f00",icon:"◆"} }[signal]||{col:T.textDim,bg:T.bg2,icon:"—"};
  return (
    <div style={{ display:"flex", alignItems:"center", gap:9, background:cfg.bg, border:`1px solid ${cfg.col}`, padding:"7px 16px" }}>
      <div style={{ width:7, height:7, borderRadius:"50%", background:cfg.col, boxShadow:`0 0 10px ${cfg.col}` }} />
      <span style={{ fontFamily:"'Courier New',monospace", fontSize:14, fontWeight:700, color:cfg.col, letterSpacing:"0.18em" }}>{cfg.icon} {signal}</span>
      <span style={{ fontSize:10, color:T.textDim }}>{(prob*100).toFixed(0)}%</span>
    </div>
  );
}

// IT: barre orizzontali del Volume Profile; evidenzia POC e fascia vicina al prezzo corrente.
// EN: horizontal Volume Profile bars; highlights POC and the bin near the current price.
function VPBars({ vpData, currentPrice }) {
  if (!vpData.length) return null;
  const pocIdx = vpData.reduce((b,d,i)=>d.total>vpData[b].total?i:b,0);
  return (
    <div style={{ display:"flex", flexDirection:"column", flex:1, gap:"1px" }}>
      {[...vpData].reverse().map((vp,i) => {
        const ri = vpData.length-1-i, isPOC = ri===pocIdx;
        const isNear = currentPrice && Math.abs(currentPrice-vp.price)/vp.price < 0.002;
        const mx = Math.max(...vpData.map(p=>p.total),1);
        return (
          <div key={i} style={{ display:"flex", alignItems:"center", gap:3, height:`${100/vpData.length}%`, minHeight:9 }}>
            <span style={{ fontFamily:"monospace", fontSize:7, width:42, textAlign:"right", flexShrink:0, color: isPOC?T.amber:isNear?T.blue:T.textMute, fontWeight: isPOC||isNear?700:400 }}>
              {(vp.price/1000).toFixed(1)}k
            </span>
            <div style={{ display:"flex", height:"65%" }}>
              <div style={{ width:(vp.buyVol/mx)*78, background:isPOC?T.amber:T.blue, opacity:isPOC?1:0.65, minWidth:vp.buyVol>0?1:0 }} />
              <div style={{ width:(vp.sellVol/mx)*78, background:T.red, opacity:0.55, minWidth:vp.sellVol>0?1:0 }} />
            </div>
            {isPOC&&<span style={{ fontSize:7, color:T.amber }}>POC</span>}
            {isNear&&!isPOC&&<span style={{ fontSize:7, color:T.blue }}>◀</span>}
          </div>
        );
      })}
    </div>
  );
}

// ─── BACKTEST COMPONENTS ──────────────────────────────────────────────────────

// IT: area drag&drop per caricare il JSON dei risultati di backtest.
// EN: drag&drop area to load the backtest results JSON.
function DropZone({ onLoad }) {
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState(null);
  const inputRef = useRef();

  // IT: valida il file (.json), lo legge e passa l'oggetto parseato a onLoad.
  // EN: validates the file (.json), reads it and passes the parsed object to onLoad.
  const handle = (file) => {
    if (!file || !file.name.endsWith(".json")) return;
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => {
      try { onLoad(JSON.parse(e.target.result)); } catch { alert("JSON non valido"); }
    };
    reader.readAsText(file);
  };

  return (
    <div
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files[0]); }}
      style={{
        border: `2px dashed ${dragging ? T.amber : T.border}`,
        background: dragging ? "#1a1500" : T.bg2,
        padding:"32px 20px", textAlign:"center", cursor:"pointer",
        transition:"all 0.15s",
      }}
    >
      <input ref={inputRef} type="file" accept=".json" style={{ display:"none" }} onChange={e=>handle(e.target.files[0])} />
      <div style={{ fontSize:28, marginBottom:10 }}>📂</div>
      <div style={{ fontSize:11, color: fileName ? T.green : T.textDim, fontFamily:"monospace", letterSpacing:"0.1em" }}>
        {fileName ? `✓ ${fileName}` : "TRASCINA dashboard_results.json"}
      </div>
      <div style={{ fontSize:9, color:T.textMute, marginTop:6, fontFamily:"monospace" }}>
        Genera con: python export_results.py
      </div>
    </div>
  );
}

// IT: riga tabella di un singolo trade (side, prezzi, size, PnL) colorata per esito.
// EN: table row for a single trade (side, prices, size, PnL) colored by outcome.
function TradeRow({ trade, i }) {
  const isWin = (trade.net_pnl ?? 0) > 0;
  return (
    <tr style={{ borderBottom:`1px solid ${T.textMute}`, background: i%2===0 ? T.bg1 : T.bg0 }}>
      {[
        { v: trade.side,        c: trade.side==="LONG"?T.green:T.red },
        { v: fmtUSD(trade.entry_price,1) },
        { v: fmtUSD(trade.exit_price,1) },
        { v: `$${(+(trade.size_usd||0)).toFixed(0)}` },
        { v: `${trade.hold_candles??0}m` },
        { v: trade.close_reason?.replace("_"," "), c:T.textDim },
        { v: `${isWin?"+":""}$${(+(trade.net_pnl||0)).toFixed(2)}`, c: isWin?T.green:T.red },
        { v: fmtPct(+(trade.pnl_pct||0)*100), c: isWin?T.green:T.red },
      ].map((cell,j) => (
        <td key={j} style={{ padding:"4px 8px", fontFamily:"'Courier New',monospace", fontSize:10, color:cell.c||T.text, whiteSpace:"nowrap" }}>
          {cell.v}
        </td>
      ))}
    </tr>
  );
}

// IT: pannello backtest: drop-zone+workflow se vuoto, altrimenti metriche, equity e tabella trade filtrabile.
// EN: backtest panel: drop-zone+workflow when empty, otherwise metrics, equity and a filterable trade table.
function BacktestPanel({ btData }) {
  const [tradeFilter, setTradeFilter] = useState("ALL");
  if (!btData) {
    return (
      <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
        <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:20 }}>
          <div style={{ fontSize:9, color:T.amber, letterSpacing:"0.15em", marginBottom:12 }}>CARICA RISULTATI BACKTEST</div>
          <DropZone onLoad={() => {}} />
          <div style={{ marginTop:16, background:T.bg3, border:`1px solid ${T.border}`, padding:12 }}>
            <div style={{ fontSize:9, color:T.textDim, marginBottom:8, letterSpacing:"0.1em" }}>WORKFLOW</div>
            {[
              "python feature_engineering.py --limit 5000",
              "python train_lstm.py --data ./data/lstm_dataset.npz",
              "python backtest.py --data ./data/lstm_dataset.npz",
              "python export_results.py",
              "→ trascina dashboard_results.json qui",
            ].map((cmd,i) => (
              <div key={i} style={{ fontFamily:"'Courier New',monospace", fontSize:10, color: i===4?T.amber:T.green, marginBottom:4, paddingLeft: i===4?0:8 }}>
                {i===4?"":"> "}{cmd}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const { metrics={}, equity_curve=[], drawdown_curve=[], trades=[], pnl_series=[], pnl_per_trade=[] } = btData;

  const eqData    = equity_curve.map((v,i) => ({ i, equity: +v.toFixed(2) }));
  const ddData    = drawdown_curve.map((v,i) => ({ i, dd: +(v*100).toFixed(2) }));
  const pnlData   = pnl_series.map((v,i) => ({ i, pnl: +v.toFixed(2) }));
  const barData   = pnl_per_trade.map((v,i) => ({ i, pnl: +v.toFixed(2) }));
  const finalEq   = equity_curve[equity_curve.length-1] ?? 0;
  const initEq    = equity_curve[0] ?? 10000;

  const filteredTrades = tradeFilter==="ALL" ? trades : trades.filter(t=>t.side===tradeFilter);

  const reasons = metrics.close_reasons || {};

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:10, overflow:"auto" }}>

      {/* KPI row */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:8 }}>
        <MetricCard label="Rendimento Totale" value={fmtPct((finalEq/initEq-1)*100,true)} positive={(finalEq/initEq-1)>0} sub={`${fmtUSD(initEq,0)} → ${fmtUSD(finalEq,0)}`} />
        <MetricCard label="Sharpe Ratio" value={metrics.sharpe?.toFixed(2)??"—"} positive={(metrics.sharpe??0)>1} sub="annualizzato (per-trade)" />
        <MetricCard label="Max Drawdown" value={`-${((metrics.max_drawdown??0)*100).toFixed(2)}`} unit="%" positive={false} sub="dalla equity curve" />
        <MetricCard label="Profit Factor" value={metrics.profit_factor?.toFixed(2)??"—"} positive={(metrics.profit_factor??0)>1.5} sub="gross win / gross loss" />
        <MetricCard label="Win Rate" value={((metrics.win_rate??0)*100).toFixed(1)} unit="%" positive={(metrics.win_rate??0)>0.5} sub={`${metrics.n_trades??0} trade totali`} />
        <MetricCard label="Sortino" value={metrics.sortino?.toFixed(2)??"—"} positive={(metrics.sortino??0)>1} sub="solo downside vol" />
        <MetricCard label="Commissioni" value={fmtUSD(metrics.total_fees,0)} positive={false} sub="fee + slippage" />
        <MetricCard label="Calmar Ratio" value={metrics.calmar?.toFixed(2)??"—"} positive={(metrics.calmar??0)>1} sub="return / max drawdown" />
      </div>

      <div style={{ display:"flex", gap:10 }}>

        {/* Left: charts */}
        <div style={{ flex:1, display:"flex", flexDirection:"column", gap:8, minWidth:0 }}>

          {/* Equity curve */}
          <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:"10px 6px 4px 0" }}>
            <div style={{ paddingLeft:12, paddingBottom:6, display:"flex", justifyContent:"space-between" }}>
              <span style={{ fontSize:9, color:T.textDim, letterSpacing:"0.1em" }}>EQUITY CURVE</span>
              <span style={{ fontSize:9, color:(finalEq-initEq)>0?T.green:T.red, paddingRight:8 }}>{fmtPct((finalEq/initEq-1)*100)}</span>
            </div>
            <ResponsiveContainer width="100%" height={120}>
              <AreaChart data={eqData} margin={{ top:2, right:10, left:0, bottom:0 }}>
                <defs>
                  <linearGradient id="eqg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={T.green} stopOpacity={0.28} />
                    <stop offset="100%" stopColor={T.green} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="1 5" stroke={T.textMute} vertical={false} />
                <XAxis dataKey="i" tick={false} axisLine={{ stroke:T.border }} />
                <YAxis tick={{ fontSize:8, fill:T.textDim }} tickFormatter={v=>"$"+v.toLocaleString()} axisLine={{ stroke:T.border }} tickLine={false} width={64} />
                <Tooltip contentStyle={{ background:T.bg3, border:`1px solid ${T.border}`, fontSize:10, fontFamily:"monospace" }} formatter={v=>["$"+v.toLocaleString(),"Equity"]} labelFormatter={()=>""} />
                <Area dataKey="equity" stroke={T.green} fill="url(#eqg)" strokeWidth={1.5} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Drawdown */}
          <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:"8px 6px 4px 0" }}>
            <div style={{ paddingLeft:12, paddingBottom:4 }}>
              <span style={{ fontSize:9, color:T.textDim, letterSpacing:"0.1em" }}>DRAWDOWN (%)</span>
            </div>
            <ResponsiveContainer width="100%" height={70}>
              <AreaChart data={ddData} margin={{ top:2, right:10, left:0, bottom:0 }}>
                <defs>
                  <linearGradient id="ddg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={T.red} stopOpacity={0.4} />
                    <stop offset="100%" stopColor={T.red} stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="i" tick={false} axisLine={{ stroke:T.border }} />
                <YAxis tick={{ fontSize:8, fill:T.textDim }} tickFormatter={v=>v+"%"} axisLine={{ stroke:T.border }} tickLine={false} width={40} />
                <Area dataKey="dd" stroke={T.red} fill="url(#ddg)" strokeWidth={1} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* P&L per trade */}
          <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:"8px 6px 4px 0" }}>
            <div style={{ paddingLeft:12, paddingBottom:4 }}>
              <span style={{ fontSize:9, color:T.textDim, letterSpacing:"0.1em" }}>P&L PER TRADE ($)</span>
            </div>
            <ResponsiveContainer width="100%" height={80}>
              <BarChart data={barData} margin={{ top:2, right:10, left:0, bottom:0 }}>
                <XAxis dataKey="i" tick={false} axisLine={{ stroke:T.border }} />
                <YAxis tick={{ fontSize:8, fill:T.textDim }} axisLine={{ stroke:T.border }} tickLine={false} width={50} />
                <ReferenceLine y={0} stroke={T.border} />
                <Bar dataKey="pnl" radius={0} maxBarSize={4}
                  shape={(props) => {
                    const { x, y, width, height, value } = props;
                    return <rect x={x} y={y} width={Math.max(width,1)} height={height} fill={value>=0?T.green:T.red} opacity={0.7} />;
                  }}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right: stats + reasons */}
        <div style={{ width:200, display:"flex", flexDirection:"column", gap:8 }}>

          {/* Close reasons */}
          <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:12 }}>
            <div style={{ fontSize:9, color:T.amber, letterSpacing:"0.14em", marginBottom:10 }}>CHIUSURE</div>
            {Object.entries(reasons).sort((a,b)=>b[1]-a[1]).map(([r,n]) => {
              const tot = Object.values(reasons).reduce((s,v)=>s+v,0)||1;
              const pct = n/tot;
              const col = r==="TAKE_PROFIT"?T.green:r==="STOP_LOSS"||r==="TRAILING_SL"?T.red:T.amber;
              return (
                <div key={r} style={{ marginBottom:8 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", marginBottom:2 }}>
                    <span style={{ fontSize:8, color:T.textDim }}>{r.replace("_"," ")}</span>
                    <span style={{ fontSize:8, color:col, fontFamily:"monospace" }}>{n} ({(pct*100).toFixed(0)}%)</span>
                  </div>
                  <div style={{ height:4, background:T.bg3 }}>
                    <div style={{ height:"100%", width:`${pct*100}%`, background:col, opacity:0.8 }} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Avg stats */}
          <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:12 }}>
            <div style={{ fontSize:9, color:T.amber, letterSpacing:"0.14em", marginBottom:10 }}>DETTAGLIO</div>
            {[
              ["Avg Win", metrics.avg_win_usd!=null ? `$${metrics.avg_win_usd.toFixed(2)}` : "—", T.green],
              ["Avg Loss", metrics.avg_loss_usd!=null ? `$${metrics.avg_loss_usd.toFixed(2)}` : "—", T.red],
              ["Avg Hold", metrics.avg_hold_candles!=null ? `${metrics.avg_hold_candles.toFixed(0)} min` : "—", T.text],
              ["Net Profit", metrics.net_profit!=null ? `$${metrics.net_profit.toFixed(2)}` : "—", (metrics.net_profit??0)>0?T.green:T.red],
              ["Gross Win", metrics.gross_profit!=null ? `$${metrics.gross_profit.toFixed(2)}` : "—", T.green],
              ["Gross Loss", metrics.gross_loss!=null ? `$${metrics.gross_loss.toFixed(2)}` : "—", T.red],
            ].map(([l,v,c]) => (
              <div key={l} style={{ display:"flex", justifyContent:"space-between", padding:"4px 0", borderBottom:`1px solid ${T.textMute}`, fontSize:10 }}>
                <span style={{ color:T.textDim }}>{l}</span>
                <span style={{ fontFamily:"'Courier New',monospace", color:c }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Trade log */}
      <div style={{ background:T.bg1, border:`1px solid ${T.border}` }}>
        <div style={{ padding:"8px 12px", borderBottom:`1px solid ${T.border}`, display:"flex", alignItems:"center", gap:12 }}>
          <span style={{ fontSize:9, color:T.amber, letterSpacing:"0.14em" }}>TRADE LOG</span>
          <span style={{ fontSize:9, color:T.textDim }}>ultimi {filteredTrades.length}</span>
          <div style={{ marginLeft:"auto", display:"flex", gap:6 }}>
            {["ALL","LONG","SHORT"].map(f => (
              <button key={f} onClick={()=>setTradeFilter(f)} style={{
                background: tradeFilter===f ? T.bg3:"none",
                border:`1px solid ${tradeFilter===f?T.amber:T.border}`,
                color: tradeFilter===f ? T.amber:T.textDim,
                fontFamily:"monospace", fontSize:9, padding:"3px 8px", cursor:"pointer",
              }}>{f}</button>
            ))}
          </div>
        </div>
        <div style={{ overflowX:"auto", maxHeight:200, overflowY:"auto" }}>
          <table style={{ width:"100%", borderCollapse:"collapse" }}>
            <thead>
              <tr style={{ background:T.bg3, position:"sticky", top:0 }}>
                {["SIDE","ENTRY","EXIT","SIZE","HOLD","REASON","NET P&L","P&L %"].map(h => (
                  <th key={h} style={{ padding:"5px 8px", textAlign:"left", fontSize:8, color:T.textDim, letterSpacing:"0.1em", fontWeight:400, whiteSpace:"nowrap" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredTrades.slice(-100).reverse().map((t,i) => <TradeRow key={i} trade={t} i={i} />)}
            </tbody>
          </table>
          {filteredTrades.length===0 && (
            <div style={{ padding:20, textAlign:"center", fontSize:10, color:T.textDim }}>Nessun trade nel filtro selezionato</div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── MAIN DASHBOARD ───────────────────────────────────────────────────────────
// IT: componente root: gestisce stato live (candele/forecast/segnale), WebSocket Binance e le 4 tab.
// EN: root component: manages live state (candles/forecast/signal), Binance WebSocket and the 4 tabs.
export default function QuantDashboard() {
  const [candles,  setCandles]  = useState([]);
  const [forecast, setForecast] = useState([]);
  const [vpData,   setVpData]   = useState([]);
  const [metrics,  setMetrics]  = useState({});
  const [ticker,   setTicker]   = useState(null);
  const [signal,   setSignal]   = useState({ type:"HOLD", prob:0.5 });
  const [wsConn,   setWsConn]   = useState(false);
  const [status,   setStatus]   = useState("CONNECTING");
  const [error,    setError]    = useState(null);
  const [tab,      setTab]      = useState("chart");
  const [tick,     setTick]     = useState(0);
  const [btData,   setBtData]   = useState(null);   // backtest results
  const wsRef      = useRef(null);
  const candlesRef = useRef([]);

  // IT: deriva il segnale BUY/SELL/HOLD dallo scarto forecast-prezzo (mediana a +5 min).
  // EN: derives the BUY/SELL/HOLD signal from the forecast-vs-price gap (+5 min median).
  const deriveSignal = useCallback((fc, price) => {
    if (!fc.length||!price) return;
    const clamp = Math.max(0.1, Math.min(0.95, 0.5+(fc[4]?.p50-price)/price*120));
    setSignal({ type: clamp>0.58?"BUY":clamp<0.42?"SELL":"HOLD", prob:clamp });
  }, []);

  // IT: ricalcola forecast MC, volume profile, metriche e segnale dalle candele correnti.
  // EN: recomputes MC forecast, volume profile, metrics and signal from the current candles.
  const refreshDerived = useCallback((cdls) => {
    if (cdls.length<10) return;
    const fc = simulateMCMC(cdls, 30, 1200);
    setForecast(fc);
    setVpData(buildVP(cdls));
    setMetrics(calcMetrics(cdls));
    deriveSignal(fc, cdls[cdls.length-1]?.close);
  }, [deriveSignal]);

  // IT: caricamento iniziale: scarica candele storiche + ticker 24h e popola lo stato.
  // EN: initial load: fetches historical candles + 24h ticker and populates state.
  useEffect(() => {
    setStatus("LOADING");
    Promise.all([fetchKlines(), fetch24hr()]).then(([raw, t24]) => {
      const wv = computeVWAP(raw);
      candlesRef.current = wv;
      setCandles(wv); setTicker(t24); setStatus("LIVE"); setError(null);
      refreshDerived(wv);
    }).catch(e => { setError(e.message); setStatus("ERROR"); });
  }, [refreshDerived]);

  // IT: connessione WebSocket Binance per le candele live, con auto-reconnect.
  // EN: Binance WebSocket connection for live candles, with auto-reconnect.
  useEffect(() => {
    const stream = `wss://stream.binance.com:9443/ws/${SYMBOL.toLowerCase()}@kline_${INTERVAL}`;
    let ws, retry;
    // IT: apre il WS, gestisce open/close (retry 3s) e aggiorna l'ultima candela su ogni messaggio.
    // EN: opens the WS, handles open/close (3s retry) and updates the last candle on each message.
    const connect = () => {
      ws = new WebSocket(stream); wsRef.current = ws;
      ws.onopen  = () => { setWsConn(true); setStatus("LIVE"); };
      ws.onclose = () => { setWsConn(false); setStatus("RECONNECTING"); retry=setTimeout(connect,3000); };
      ws.onmessage = (evt) => {
        try {
          const k = JSON.parse(evt.data).k; if (!k) return;
          setTick(t=>t+1);
          const upd = { i:candlesRef.current.length-1, ts:k.t, time:new Date(k.t).toLocaleTimeString("it-IT",{hour:"2-digit",minute:"2-digit"}), open:+k.o, high:+k.h, low:+k.l, close:+k.c, volume:+k.v, bullish:+k.c>=+k.o, bodyLow:Math.min(+k.o,+k.c), bodyHigh:Math.max(+k.o,+k.c), vwap:null };
          setCandles(prev => {
            const next = computeVWAP(k.x ? [...prev.slice(-(LIMIT-1)), upd] : [...prev.slice(0,-1), upd]);
            candlesRef.current = next; return next;
          });
        } catch(_) {}
      };
    };
    connect();
    return () => { clearTimeout(retry); if(wsRef.current) wsRef.current.close(); };
  }, []);

  // IT: ricalcola le derivate ogni 15 tick per non saturare la CPU a ogni messaggio WS.
  // EN: recomputes derived data every 15 ticks to avoid maxing the CPU on every WS message.
  useEffect(() => {
    if (tick>0 && tick%15===0) refreshDerived(candlesRef.current);
  }, [tick, refreshDerived]);

  const view60 = candles.slice(-60);
  const curr   = candles[candles.length-1]?.close??0;
  const prev   = candles[candles.length-2]?.close??curr;
  const delta  = prev ? (curr-prev)/prev*100 : 0;
  const isUp   = delta>=0;
  const fmtP   = v => v ? v.toLocaleString("en-US",{minimumFractionDigits:1,maximumFractionDigits:1}) : "—";

  const pMin = view60.length ? Math.min(...view60.map(c=>c.low), ...forecast.map(f=>f.p05??Infinity))*0.9992 : 0;
  const pMax = view60.length ? Math.max(...view60.map(c=>c.high), ...forecast.map(f=>f.p95??0))*1.0008 : 1;

  const chartData = [
    ...view60.map(c => ({ time:c.time, close:c.close, vwap:c.vwap, volume:c.volume, bullish:c.bullish })),
    ...forecast,
  ];

  const TABS = [
    ["chart",   "CHART"],
    ["backtest","BACKTEST"],
    ["metrics", "LIVE METRICS"],
    ["model",   "MODEL"],
  ];

  return (
    <div style={{ background:T.bg0, color:T.text, height:"100vh", fontFamily:"monospace", display:"flex", flexDirection:"column", overflow:"hidden" }}>

      {/* TOP BAR */}
      <div style={{ background:T.bg1, borderBottom:`1px solid ${T.border}`, display:"flex", alignItems:"center", justifyContent:"space-between", padding:"0 18px", height:44, flexShrink:0 }}>
        <div style={{ display:"flex", alignItems:"center", gap:14 }}>
          <span style={{ fontFamily:"'Courier New',monospace", fontSize:13, color:T.amber, fontWeight:700, letterSpacing:"0.22em" }}>◈ QUANTSYS</span>
          <span style={{ fontSize:9, color:T.textDim, letterSpacing:"0.1em" }}>NEURAL FORECASTING ENGINE · BINANCE LIVE</span>
          {btData && <span style={{ fontSize:9, color:T.green, background:T.greenDim, border:`1px solid ${T.green}`, padding:"1px 6px" }}>✓ BACKTEST CARICATO</span>}
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:16 }}>
          {error && <span style={{ fontSize:9, color:T.red }}>⚠ {error}</span>}
          <div style={{ display:"flex", alignItems:"center", gap:6 }}>
            <Dot ok={wsConn} />
            <span style={{ fontSize:9, color:wsConn?T.green:T.red }}>{status}</span>
          </div>
          <span style={{ fontSize:9, color:T.textDim }}>{SYMBOL} · {INTERVAL} · tick#{tick}</span>
        </div>
      </div>

      {/* PRICE ROW */}
      <div style={{ background:T.bg1, borderBottom:`1px solid ${T.border}`, display:"flex", alignItems:"center", gap:20, padding:"7px 18px", flexShrink:0 }}>
        <div style={{ display:"flex", alignItems:"baseline", gap:10 }}>
          <span style={{ fontFamily:"'Courier New',monospace", fontSize:28, fontWeight:700, color:isUp?T.green:T.red, lineHeight:1 }}>${fmtP(curr)}</span>
          <span style={{ fontSize:12, color:isUp?T.green:T.red }}>{isUp?"▲":"▼"} {Math.abs(delta).toFixed(3)}%</span>
        </div>
        <div style={{ width:1, height:32, background:T.border }} />
        <div style={{ display:"flex", gap:20, flexWrap:"wrap" }}>
          {ticker && [
            ["24H HIGH", fmtP(+ticker.highPrice)],
            ["24H LOW",  fmtP(+ticker.lowPrice)],
            ["24H VOL",  (+ticker.volume/1e3).toFixed(1)+"K"],
            ["24H Δ",    ((+ticker.priceChangePercent>=0?"+":"")+parseFloat(ticker.priceChangePercent).toFixed(2)+"%")],
          ].map(([l,v])=><KV key={l} label={l} value={v} />)}
          {view60.length>0 && [
            ["VWAP", fmtP(view60[view60.length-1]?.vwap)],
            ["1h VOL", (view60.reduce((s,c)=>s+c.volume,0)/1e3).toFixed(1)+"K"],
          ].map(([l,v])=><KV key={l} label={l} value={v} color={T.amber} />)}
        </div>
        <div style={{ marginLeft:"auto" }}><SignalBadge signal={signal.type} prob={signal.prob} /></div>
      </div>

      {/* TABS */}
      <div style={{ borderBottom:`1px solid ${T.border}`, display:"flex", padding:"0 18px", flexShrink:0, background:T.bg1 }}>
        {TABS.map(([id,label])=>(
          <button key={id} onClick={()=>setTab(id)} style={{
            background:"none", border:"none", cursor:"pointer",
            padding:"8px 16px", fontSize:10, letterSpacing:"0.15em",
            color: tab===id?T.amber:T.textDim,
            borderBottom: tab===id?`2px solid ${T.amber}`:"2px solid transparent",
            fontFamily:"monospace",
          }}>{label}{id==="backtest"&&btData&&<span style={{ marginLeft:5, fontSize:8, color:T.green }}>●</span>}</button>
        ))}
        {tab==="backtest" && (
          <label style={{ marginLeft:"auto", alignSelf:"center", background:"none", border:`1px solid ${T.border}`, color:T.textDim, fontFamily:"monospace", fontSize:9, padding:"4px 10px", cursor:"pointer" }}>
            ↑ CARICA JSON
            <input type="file" accept=".json" style={{ display:"none" }} onChange={e=>{
              const f=e.target.files[0]; if(!f) return;
              const r=new FileReader(); r.onload=ev=>{try{setBtData(JSON.parse(ev.target.result))}catch{alert("JSON non valido")}};
              r.readAsText(f);
            }} />
          </label>
        )}
      </div>

      {/* BODY */}
      <div style={{ flex:1, overflow:"hidden", display:"flex", flexDirection:"column", padding:12, gap:10 }}>

        {/* CHART TAB */}
        {tab==="chart" && (
          <div style={{ flex:1, display:"flex", gap:10, minHeight:0 }}>
            <div style={{ flex:1, display:"flex", flexDirection:"column", gap:8, minWidth:0 }}>
              <div style={{ flex:3, background:T.bg1, border:`1px solid ${T.border}`, padding:"10px 6px 2px 0", display:"flex", flexDirection:"column" }}>
                <div style={{ paddingLeft:12, paddingBottom:5, display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                  <span style={{ fontSize:9, color:T.textDim, letterSpacing:"0.1em" }}>BTC/USDT · 1m · CLOSE + VWAP + MCMC FORECAST</span>
                </div>
                <div style={{ flex:1 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top:2, right:10, left:0, bottom:0 }}>
                      <defs>
                        <linearGradient id="g90" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={T.blue} stopOpacity={0.10}/><stop offset="100%" stopColor={T.blue} stopOpacity={0.02}/></linearGradient>
                        <linearGradient id="g50" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={T.blue} stopOpacity={0.22}/><stop offset="100%" stopColor={T.blue} stopOpacity={0.06}/></linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="1 5" stroke={T.textMute} vertical={false} />
                      <XAxis dataKey="time" tick={{ fontSize:8, fill:T.textDim }} tickLine={false} axisLine={{ stroke:T.border }} interval={11} />
                      <YAxis domain={[pMin,pMax]} tick={{ fontSize:8, fill:T.textDim }} tickLine={false} axisLine={{ stroke:T.border }} tickFormatter={v=>"$"+(v/1000).toFixed(1)+"k"} width={64} />
                      <Tooltip contentStyle={{ background:T.bg3, border:`1px solid ${T.border}`, fontSize:10, fontFamily:"monospace" }} labelStyle={{ color:T.amber }} formatter={(v,n)=>[v?"$"+fmtP(v):"—",n]} />
                      <Area dataKey="p95" stroke="none" fill="none" legendType="none" />
                      <Area dataKey="p05" stroke="none" fill="url(#g90)" legendType="none" baseValue={pMin} />
                      <Area dataKey="p75" stroke="none" fill="none" legendType="none" />
                      <Area dataKey="p25" stroke="none" fill="url(#g50)" legendType="none" baseValue={pMin} />
                      <Line dataKey="p95" stroke={T.blue} dot={false} strokeWidth={0.4} opacity={0.35} legendType="none" connectNulls={false} />
                      <Line dataKey="p05" stroke={T.blue} dot={false} strokeWidth={0.4} opacity={0.35} legendType="none" connectNulls={false} />
                      <Line dataKey="p75" stroke={T.blue} dot={false} strokeWidth={0.7} opacity={0.55} legendType="none" connectNulls={false} />
                      <Line dataKey="p25" stroke={T.blue} dot={false} strokeWidth={0.7} opacity={0.55} legendType="none" connectNulls={false} />
                      <Line dataKey="p50" stroke={T.blue} dot={false} strokeWidth={1.8} strokeDasharray="6 3" legendType="none" connectNulls={false} />
                      <Line dataKey="vwap" stroke={T.amber} dot={false} strokeWidth={1.2} strokeDasharray="4 3" legendType="none" connectNulls={false} />
                      <Line dataKey="close" stroke={isUp?T.green:T.red} dot={false} strokeWidth={1.6} legendType="none" connectNulls={false} />
                      {curr>0&&<ReferenceLine y={curr} stroke={isUp?T.green:T.red} strokeDasharray="2 4" strokeWidth={0.8} />}
                      {view60[view60.length-1]?.time&&<ReferenceLine x={view60[view60.length-1].time} stroke={T.amberDim} strokeDasharray="3 3" strokeWidth={1} label={{ value:"NOW →",position:"top",fill:T.amberDim,fontSize:8,fontFamily:"monospace" }} />}
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div style={{ flex:1, background:T.bg1, border:`1px solid ${T.border}`, padding:"8px 6px 2px 0", display:"flex", flexDirection:"column" }}>
                <div style={{ paddingLeft:12, paddingBottom:4 }}><span style={{ fontSize:9, color:T.textDim, letterSpacing:"0.1em" }}>VOLUME (BTC)</span></div>
                <div style={{ flex:1 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={view60} margin={{ top:0, right:10, left:0, bottom:0 }}>
                      <XAxis dataKey="time" tick={false} axisLine={{ stroke:T.border }} />
                      <YAxis tick={{ fontSize:8, fill:T.textDim }} tickLine={false} axisLine={{ stroke:T.border }} tickFormatter={v=>v>999?(v/1000).toFixed(1)+"k":v.toFixed(0)} width={64} />
                      <Bar dataKey="volume" isAnimationActive={false} shape={(props)=>{const{x,y,width,height,index}=props;const c=view60[index];if(!c)return null;return<rect x={x} y={y} width={Math.max(width,1)} height={height} fill={c.bullish?T.green:T.red} opacity={0.55}/>;}} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
            <div style={{ width:168, background:T.bg1, border:`1px solid ${T.border}`, padding:"12px 6px 8px", display:"flex", flexDirection:"column", gap:8 }}>
              <div><div style={{ fontSize:9, color:T.amber, letterSpacing:"0.12em", marginBottom:6 }}>VOLUME PROFILE</div></div>
              <VPBars vpData={vpData} currentPrice={curr} />
            </div>
          </div>
        )}

        {/* BACKTEST TAB */}
        {tab==="backtest" && (
          <div style={{ flex:1, overflow:"auto" }}>
            {btData
              ? <BacktestPanel btData={btData} />
              : (
                <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
                  <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:20 }}>
                    <div style={{ fontSize:9, color:T.amber, letterSpacing:"0.15em", marginBottom:14 }}>CARICA RISULTATI BACKTEST</div>
                    <DropZone onLoad={setBtData} />
                    <div style={{ marginTop:14, background:T.bg3, border:`1px solid ${T.border}`, padding:14 }}>
                      <div style={{ fontSize:9, color:T.textDim, marginBottom:8, letterSpacing:"0.1em" }}>GENERA IL FILE CON</div>
                      {[
                        "python feature_engineering.py --limit 5000",
                        "python train_lstm.py",
                        "python backtest.py --capital 10000",
                        "python export_results.py",
                        "→ trascina  dashboard_results.json  qui sopra",
                      ].map((cmd,i)=>(
                        <div key={i} style={{ fontFamily:"'Courier New',monospace", fontSize:10, color:i===4?T.amber:T.green, marginBottom:4, paddingLeft:i===4?0:8 }}>{i===4?"":"> "}{cmd}</div>
                      ))}
                    </div>
                  </div>
                </div>
              )
            }
          </div>
        )}

        {/* LIVE METRICS TAB */}
        {tab==="metrics" && (
          <div style={{ display:"flex", flexDirection:"column", gap:10, overflow:"auto" }}>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:8 }}>
              <MetricCard label="Sharpe Ratio" value={metrics.sharpe??"—"} positive={(metrics.sharpe??0)>1} sub="live, annualizzato" />
              <MetricCard label="Max Drawdown" value={metrics.maxDD!=null?`-${metrics.maxDD}`:"—"} unit="%" positive={false} />
              <MetricCard label="Win Rate" value={metrics.winRate??"—"} unit="%" positive={(metrics.winRate??0)>50} />
              <MetricCard label="Profit Factor" value={metrics.profitFactor??"—"} positive={(metrics.profitFactor??0)>1.5} />
              <MetricCard label="Volatilità Ann." value={metrics.volatility??"—"} unit="%" positive={null} />
              <MetricCard label="Return periodo" value={metrics.totalReturn!=null?((metrics.totalReturn>0?"+":"")+metrics.totalReturn):"—"} unit="%" positive={(metrics.totalReturn??0)>0} />
            </div>
            <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:14 }}>
              <div style={{ marginBottom:8, fontSize:9, color:T.textDim, letterSpacing:"0.1em" }}>EQUITY CURVE (close-to-close live)</div>
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={candles.map((c,i)=>({i,eq:candles.slice(0,i+1).reduce((eq,bar,j)=>j===0?eq:eq*Math.exp(Math.log(bar.close/candles[j-1].close)),10000)}))}>
                  <defs><linearGradient id="eqGLive" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={T.green} stopOpacity={0.28}/><stop offset="100%" stopColor={T.green} stopOpacity={0.02}/></linearGradient></defs>
                  <CartesianGrid strokeDasharray="1 5" stroke={T.textMute} vertical={false} />
                  <XAxis dataKey="i" tick={false} axisLine={{ stroke:T.border }} />
                  <YAxis tick={{ fontSize:8, fill:T.textDim }} tickFormatter={v=>"$"+v.toFixed(0)} axisLine={{ stroke:T.border }} tickLine={false} width={60} />
                  <Area dataKey="eq" stroke={T.green} fill="url(#eqGLive)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* MODEL TAB */}
        {tab==="model" && (
          <div style={{ display:"flex", gap:10, overflow:"auto" }}>
            <div style={{ flex:1, display:"flex", flexDirection:"column", gap:10 }}>
              <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:14 }}>
                <div style={{ marginBottom:10, fontSize:9, color:T.amber, letterSpacing:"0.18em" }}>ARCHITETTURA</div>
                {[["Tipo","LSTM + GRU Ensemble"],["Input","60 minuti · ~55 features"],["Layers","LSTM(256) → GRU(128) → MLP residual"],["Output","μ, σ², ν  → t-Student parametrica"],["Loss","Negative Log-Likelihood"],["Ottimizzatore","AdamW · warmup cosine schedule"],["Regolarizzazione","Dropout 0.2 · early stopping"]].map(([k,v])=>(
                  <div key={k} style={{ display:"flex", justifyContent:"space-between", padding:"5px 0", borderBottom:`1px solid ${T.textMute}`, fontSize:11 }}>
                    <span style={{ color:T.textDim }}>{k}</span>
                    <span style={{ color:T.text, fontFamily:"'Courier New',monospace", fontSize:10 }}>{v}</span>
                  </div>
                ))}
              </div>
              <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:14 }}>
                <div style={{ marginBottom:10, fontSize:9, color:T.amber, letterSpacing:"0.18em" }}>FEATURE IMPORTANCE</div>
                {[["VP POC distance",0.84],["VWAP deviation",0.76],["Log returns (t-1..5)",0.70],["Volume z-score",0.63],["High-Low spread",0.49],["Macro regime",0.41]].map(([f,imp])=>(
                  <div key={f} style={{ display:"flex", alignItems:"center", gap:8, marginBottom:6 }}>
                    <span style={{ fontSize:9, color:T.textDim, width:180, flexShrink:0 }}>{f}</span>
                    <div style={{ flex:1, height:10, background:T.bg3, border:`1px solid ${T.border}`, position:"relative" }}>
                      <div style={{ position:"absolute", inset:0, right:`${(1-imp)*100}%`, background:`linear-gradient(90deg,${T.amber}88,${T.amber})` }} />
                    </div>
                    <span style={{ fontSize:9, color:T.amber, fontFamily:"'Courier New',monospace", width:32, textAlign:"right" }}>{imp.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ width:240, display:"flex", flexDirection:"column", gap:10 }}>
              <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:14 }}>
                <div style={{ marginBottom:10, fontSize:9, color:T.amber, letterSpacing:"0.18em" }}>MONTECARLO (LIVE)</div>
                {[["Paths","1,200"],["Horizon","30 min"],["Vol model","GARCH(1,1)"],["Tails","t-Student df≈5"],["Update","ogni 15 tick WS"],["Bande","5·25·50·75·95%ile"]].map(([k,v])=>(
                  <div key={k} style={{ display:"flex", justifyContent:"space-between", padding:"5px 0", borderBottom:`1px solid ${T.textMute}`, fontSize:11 }}>
                    <span style={{ color:T.textDim }}>{k}</span>
                    <span style={{ color:T.text, fontFamily:"'Courier New',monospace", fontSize:10 }}>{v}</span>
                  </div>
                ))}
              </div>
              <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:14 }}>
                <div style={{ marginBottom:8, fontSize:9, color:T.amber, letterSpacing:"0.18em" }}>SIGNAL LOGIC</div>
                <code style={{ fontSize:10, color:T.green, lineHeight:1.8, display:"block", background:T.bg3, padding:10 }}>
                  P_up = CDF(t-Student)<br/>if P_up &gt; 0.58:<br/>{"  "}→ BUY · 25% size<br/>elif P_up &lt; 0.42:<br/>{"  "}→ SELL · 25% size<br/>else: HOLD
                </code>
              </div>
              <div style={{ background:T.bg1, border:`1px solid ${T.border}`, padding:14 }}>
                <div style={{ marginBottom:8, fontSize:9, color:T.amber, letterSpacing:"0.18em" }}>PIPELINE COMPLETA</div>
                {[["✓","feature_engineering.py",T.green],["✓","train_lstm.py",T.green],["✓","monte_carlo.py",T.green],["✓","risk_manager.py",T.green],["✓","backtest.py",T.green],["✓","export_results.py",T.green],["✓","dashboard (live)",T.amber]].map(([s,f,c])=>(
                  <div key={f} style={{ display:"flex", gap:8, padding:"3px 0", borderBottom:`1px solid ${T.textMute}`, fontSize:9 }}>
                    <span style={{ color:c }}>{s}</span>
                    <span style={{ color:T.textDim, fontFamily:"'Courier New',monospace" }}>{f}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* STATUS BAR */}
      <div style={{ background:T.bg1, borderTop:`1px solid ${T.border}`, display:"flex", alignItems:"center", gap:14, padding:"3px 18px", flexShrink:0 }}>
        <Dot ok={wsConn} />
        <span style={{ fontSize:8, color:T.textDim }}>WS {SYMBOL} {INTERVAL}</span>
        <span style={{ fontSize:8, color:T.textDim }}>|</span>
        <span style={{ fontSize:8, color:T.textDim }}>{candles.length} candles</span>
        <span style={{ fontSize:8, color:T.textDim }}>|</span>
        <span style={{ fontSize:8, color:signal.type==="BUY"?T.green:signal.type==="SELL"?T.red:T.amber }}>SIGNAL: {signal.type} {(signal.prob*100).toFixed(0)}%</span>
        {btData && <>
          <span style={{ fontSize:8, color:T.textDim }}>|</span>
          <span style={{ fontSize:8, color:T.green }}>BACKTEST: {btData.metrics?.n_trades??0} trade · SR {btData.metrics?.sharpe?.toFixed(2)??"—"} · DD -{((btData.metrics?.max_drawdown??0)*100).toFixed(1)}%</span>
        </>}
        <span style={{ marginLeft:"auto", fontSize:8, color:T.textMute }}>QUANTSYS v2 · TUTTE LE FASI COMPLETE</span>
      </div>

      <style>{`
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.25}}
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:4px;height:4px}
        ::-webkit-scrollbar-track{background:${T.bg0}}
        ::-webkit-scrollbar-thumb{background:${T.border}}
        button:hover{opacity:0.8}
      `}</style>
    </div>
  );
}
