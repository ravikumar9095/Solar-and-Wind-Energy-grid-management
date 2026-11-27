"""Website for solar and wind energy prediction : type the weather of a day, get that day's wind and solar energy for a state.
The estimate comes from a weather-based model (gradient boosting) learned from the weather, season and state in your CSV.

Put this file and Renewable_Climate_Solar_Wind_Updated.csv in the same folder (or pass --csv PATH), then:
    python future_site.py train        # once, a few minutes; needs torch, statsmodels, scikit-learn, joblib
    python future_site.py              # opens the website at http://127.0.0.1:5000   (needs flask, joblib, scikit-learn)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                                   # running inside Jupyter / Spyder console: use the current folder
    HERE = os.getcwd()
OUT = os.path.join(HERE, "future_model")
CSV_NAME = "Renewable_Climate_Solar_Wind_Updated.csv"
WEATHER = ["T2M", "RH2M", "PRECIPITATION", "WS10M", "WS50M", "ALLSKY_SFC_SW_DWN", "CLOUD_AMT"]
TARGETS = ["wind_energy", "solar_energy"]
CAL = ["doy_sin", "doy_cos", "dow_sin", "dow_cos"]
LOOKBACK, TRAIN_FRACTION = 14, 0.80


# ----------------------------------------------------------------------------------------------- data (same steps as the notebook)
def find_csv(arg=None):
    for p in (arg, os.environ.get("SWF_DATA"), os.path.join(HERE, CSV_NAME), os.path.join(HERE, "data", CSV_NAME), CSV_NAME):
        if p and os.path.exists(p):
            return p
    raise SystemExit(f"Cannot find {CSV_NAME}. Put it next to future_site.py or pass --csv PATH.")


def load_clean(path):
    df = pd.read_csv(path).rename(columns={"PRECTOTCORR": "PRECIPITATION"})
    missing = [c for c in ["date", "state_name"] + TARGETS + WEATHER if c not in df.columns]
    if missing:
        raise SystemExit(f"The CSV is missing the columns {missing}")
    d = df["date"].astype(str).str.strip()
    dt = pd.to_datetime(d, format="%d-%m-%Y", errors="coerce")
    df["date"] = pd.to_datetime(d, errors="coerce") if dt.isna().mean() > 0.5 else dt
    df["state_name"] = df["state_name"].astype(str).str.strip().str.lower()
    num = TARGETS + WEATHER
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.dropna(subset=["date", "state_name"]).groupby(["state_name", "date"], as_index=False)[num].mean()
          .sort_values(["state_name", "date"]).reset_index(drop=True))
    df = df.dropna(subset=TARGETS).copy()
    df[WEATHER] = df.groupby("state_name")[WEATHER].transform(lambda s: s.interpolate(limit_direction="both"))
    return df.dropna(subset=WEATHER).reset_index(drop=True)


def add_features(df, states=None):
    df = df.copy()
    doy, dow = df.date.dt.dayofyear, df.date.dt.dayofweek
    df["day_of_year"] = doy
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    states = list(states) if states is not None else sorted(df.state_name.unique())
    dummies = pd.DataFrame({f"state_{s}": (df.state_name == s).astype(float) for s in states})
    return pd.concat([df, dummies], axis=1), WEATHER + CAL + TARGETS + list(dummies.columns), states


def train_mask_of(df):
    m = np.zeros(len(df), dtype=bool)
    for _, g in df.groupby("state_name"):
        idx = g.sort_values("date").index.to_numpy()
        m[idx[:int(len(idx) * TRAIN_FRACTION)]] = True
    return m


def make_windows(df, X, Y, train_mask):
    from numpy.lib.stride_tricks import sliding_window_view
    xs, ys = {True: [], False: []}, {True: [], False: []}
    for _, g in df.groupby("state_name"):
        idx = g.sort_values("date").index.to_numpy()
        if len(idx) <= LOOKBACK:
            continue
        w = sliding_window_view(X[idx], LOOKBACK, axis=0)[:-1].transpose(0, 2, 1)      # window = the 14 days BEFORE the target day
        tgt = idx[LOOKBACK:]
        m = train_mask[tgt]
        for flag in (True, False):
            xs[flag].append(w[m == flag]); ys[flag].append(Y[tgt[m == flag]])
    cat = lambda parts: np.concatenate(parts).astype(np.float32)
    return cat(xs[True]), cat(ys[True]), cat(xs[False]), cat(ys[False])


# ----------------------------------------------------------------------------------------------- models
def build_lstm(n_features, hidden=64):
    from torch import nn

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden, 2)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    return LSTMForecaster()


def fit_ses(df, train_mask):
    """Per state and target: alpha fitted on the training part, level then rolled through ALL data (that level is the forecast)."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    ses, truth, pred = {}, {t: [] for t in TARGETS}, {t: [] for t in TARGETS}
    for s, g in df.groupby("state_name"):
        g = g.sort_values("date")
        tm = train_mask[g.index.to_numpy()]
        ses[s] = {}
        for t in TARGETS:
            tr, te = g.loc[tm, t].to_numpy(float), g.loc[~tm, t].to_numpy(float)
            try:
                p = SimpleExpSmoothing(tr, initialization_method="estimated").fit(optimized=True).params
                a, level = float(p["smoothing_level"]), float(p["initial_level"])
            except Exception:
                a, level = 0.30, (float(tr[0]) if len(tr) else 0.0)
            for y in tr:
                level = a * y + (1 - a) * level
            for y in te:                                   # rolling one-day-ahead on the test period (as in the notebook)
                pred[t].append(level); truth[t].append(y)
                level = a * y + (1 - a) * level
            ses[s][t] = dict(alpha=a, level=float(level))
    return ses, truth, pred


def train(csv, epochs):
    import joblib
    import torch
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(42); np.random.seed(42)

    df, feats, states = add_features(load_clean(csv))
    tm = train_mask_of(df)
    xs, ys = StandardScaler().fit(df.loc[tm, feats]), StandardScaler().fit(df.loc[tm, TARGETS])
    X, Y = xs.transform(df[feats]).astype(np.float32), ys.transform(df[TARGETS]).astype(np.float32)
    Xtr, Ytr, Xte, Yte = make_windows(df, X, Y, tm)
    print(f"[data] {len(df):,} rows, {len(states)} states | train windows {len(Xtr):,} | test windows {len(Xte):,} | features {len(feats)}")

    model = build_lstm(len(feats))
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=1e-3), nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)), batch_size=256, shuffle=True)
    for ep in range(epochs):
        model.train(); total = 0.0
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step(); total += loss.item() * len(xb)
        print(f"  epoch {ep + 1:02d}/{epochs}  training MSE {total / len(Xtr):.5f}")
    model.eval()
    with torch.no_grad():
        ps = np.vstack([model(torch.tensor(Xte[i:i + 1024])).numpy() for i in range(0, len(Xte), 1024)])
    y_true, y_pred = ys.inverse_transform(Yte), ys.inverse_transform(ps)

    print("[ses] fitting exponential smoothing per state")
    ses, s_true, s_pred = fit_ses(df, tm)
    score = lambda a, b: dict(rmse=float(np.sqrt(mean_squared_error(a, b))), r2=float(r2_score(a, b)))
    metrics = {"wind": dict(model="LSTM", **score(y_true[:, 0], y_pred[:, 0])),
               "solar": dict(model="Exponential smoothing", **score(s_true["solar_energy"], s_pred["solar_energy"]))}
    print("[weather model] energy of a day from that day's weather + season + state (used by the manual-input page)")
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xw = df[WEATHER + CAL + [f"state_{s}" for s in states]].to_numpy(float)
    wm = {t: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=42).fit(Xw[tm], df.loc[tm, t]) for t in TARGETS}
    wmet = {k: score(df.loc[~tm, t].to_numpy(), wm[t].predict(Xw[~tm])) for k, t in (("wind", "wind_energy"), ("solar", "solar_energy"))}
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, "lstm.pt"))
    joblib.dump(dict(feats=feats, states=states, x_mean=xs.mean_, x_scale=xs.scale_, y_mean=ys.mean_, y_scale=ys.scale_,
                     ses=ses, metrics=metrics, weather_models=wm, weather_metrics=wmet), os.path.join(OUT, "meta.joblib"))
    print(f"\nTest accuracy: wind LSTM RMSE {metrics['wind']['rmse']:.4f}, R2 {metrics['wind']['r2']:.4f} | "
          f"solar SES RMSE {metrics['solar']['rmse']:.4f}, R2 {metrics['solar']['r2']:.4f}")
    print(f"Weather-input model test accuracy: wind RMSE {wmet['wind']['rmse']:.4f}, R2 {wmet['wind']['r2']:.4f} | solar RMSE {wmet['solar']['rmse']:.4f}, R2 {wmet['solar']['r2']:.4f}")
    print(f"Saved to {OUT}. Now run:  python future_site.py")


# ----------------------------------------------------------------------------------------------- estimate from typed weather
def wx_row(weather, date, state, states):
    d = pd.Timestamp(date)
    cal = [np.sin(2 * np.pi * d.dayofyear / 365.25), np.cos(2 * np.pi * d.dayofyear / 365.25),
           np.sin(2 * np.pi * d.dayofweek / 7), np.cos(2 * np.pi * d.dayofweek / 7)]
    return np.array([float(weather[k]) for k in WEATHER] + cal + [1.0 if s == state else 0.0 for s in states])


def estimate(state, date, weather, ctx):
    """Wind and solar energy of ONE day from the weather typed by the user."""
    m = ctx["meta"]
    if "weather_models" not in m:
        raise ValueError("Run 'python future_site.py train' again to enable manual weather input.")
    try:
        row = wx_row(weather or {}, date, state, m["states"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Enter a number in every weather box and choose a valid date.")
    if not np.isfinite(row).all():
        raise ValueError("Enter a number in every weather box.")
    warn = [f"{k} = {weather[k]} is outside the range in your data ({ctx['rng'].loc['min', k]:.1f} to {ctx['rng'].loc['max', k]:.1f}); the estimate may be unreliable."
            for k in WEATHER if not ctx["rng"].loc["min", k] <= float(weather[k]) <= ctx["rng"].loc["max", k]]
    return dict(wind=max(float(m["weather_models"]["wind_energy"].predict(row[None])[0]), 0.0),
                solar=max(float(m["weather_models"]["solar_energy"].predict(row[None])[0]), 0.0), warnings=warn)


def make_ctx(df, meta):
    return dict(df=df, meta=meta, rng=df[WEATHER].agg(["min", "max"]), defaults=df[WEATHER].median().round(2).to_dict(),
                top=df.groupby("state_name")[TARGETS].max())


def create_app(ctx):
    from flask import Flask, Response, jsonify, request
    app = Flask(__name__)
    known = [s for s in sorted(ctx["df"].state_name.unique()) if s in ctx["meta"]["states"]]

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/api/info")
    def info():
        return jsonify(states=known, defaults=ctx["defaults"], ranges={k: dict(min=float(ctx["rng"].loc["min", k]), max=float(ctx["rng"].loc["max", k])) for k in WEATHER},
                       top={s: dict(wind=float(ctx["top"].loc[s, "wind_energy"]), solar=float(ctx["top"].loc[s, "solar_energy"])) for s in known},
                       weather_metrics=ctx["meta"].get("weather_metrics"))

    @app.post("/api/estimate")
    def api_estimate():
        q = request.get_json(force=True, silent=True) or {}
        state = str(q.get("state", "")).lower().strip()
        if state not in known:
            return jsonify(error=f"Unknown state '{state}'."), 400
        try:
            return jsonify(estimate(state, q.get("date"), q.get("weather"), ctx))
        except Exception as e:
            return jsonify(error=str(e)), 400
    return app


def serve(csv, port):
    import joblib
    try:
        meta = joblib.load(os.path.join(OUT, "meta.joblib"))
    except FileNotFoundError:
        raise SystemExit("No trained model found. Run first:  python future_site.py train")
    df, _, _ = add_features(load_clean(csv), meta["states"])
    print(f"Open http://127.0.0.1:{port}")
    create_app(make_ctx(df, meta)).run(port=port, debug=False)


# ----------------------------------------------------------------------------------------------- page
PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Solar and Wind Energy prediction</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;600;800&display=swap">
<style>
:root{--sky1:#cfe6f5;--sky2:#eef6ee;--panel:rgba(255,255,255,.86);--ink:#12222b;--mute:#5a6b75;--line:#d4dfe4;--field:#f4f8fa;--wind:#0b7285;--solar:#c97a00;--accent:#0b7285;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--sky1:#0d1f2d;--sky2:#101c1a;--panel:rgba(22,36,40,.88);--ink:#e8f1f4;--mute:#9db2ba;--line:#2b4047;--field:#112027;--wind:#46c7dc;--solar:#f5b942;--accent:#46c7dc;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;color:var(--ink);font:16px/1.5 "Bricolage Grotesque",system-ui,sans-serif;background:linear-gradient(180deg,var(--sky1) 0,var(--sky2) 420px) no-repeat,var(--sky2)}
main{max-width:1040px;margin:0 auto;padding:36px 20px 60px}
header{display:flex;align-items:center;gap:16px;margin-bottom:8px}
header svg{flex:none}
h1{font-size:clamp(28px,5vw,40px);line-height:1.05;margin:0;font-weight:800;letter-spacing:-.02em}
.lead{color:var(--mute);margin:8px 0 26px;max-width:64ch}
.grid{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,.9fr);gap:22px;align-items:start}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);backdrop-filter:blur(6px);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:0 8px 28px rgba(10,40,60,.08)}
.res{position:sticky;top:16px}
h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute);margin:0 0 14px;font-weight:600}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
label{display:block;font-size:14px;font-weight:600;margin:0 0 5px}
input[type=number],input[type=date],select{width:100%;padding:10px 11px;font:inherit;color:var(--ink);background:var(--field);border:1px solid var(--line);border-radius:9px}
.wx{margin-top:22px;display:grid;gap:14px}
.row .top{display:flex;justify-content:space-between;align-items:center;gap:10px}
.row .top label{margin:0;display:flex;align-items:center;gap:8px}
.row .top label i{font-style:normal;font-size:18px;width:24px;text-align:center}
.val{display:flex;align-items:center;gap:6px}.val input{width:92px;text-align:right;padding:6px 8px}.val em{font-style:normal;font-size:12px;color:var(--mute);min-width:52px}
input[type=range]{width:100%;margin:6px 0 0;accent-color:var(--accent)}
.rg{display:flex;justify-content:space-between;font-size:11px;color:var(--mute)}
.btns{display:flex;gap:10px;margin-top:22px}
button{flex:1;padding:11px;font:inherit;font-weight:600;border-radius:9px;cursor:pointer;border:1px solid var(--line);background:var(--field);color:var(--ink)}
:focus-visible{outline:3px solid var(--solar);outline-offset:2px}
.cond{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 18px}
.chip{padding:5px 11px;border-radius:999px;font-size:13px;font-weight:600;background:var(--field);border:1px solid var(--line)}
.meter{padding:16px 0;border-top:1px solid var(--line)}.meter:first-of-type{border-top:0;padding-top:0}
.mh{display:flex;align-items:center;gap:12px}.mh svg{flex:none}
.mh b{display:block;font-size:13px;font-weight:600;color:var(--mute)}
.big{font-size:38px;line-height:1;font-weight:800;letter-spacing:-.02em}.big small{font-size:14px;font-weight:600;color:var(--mute);margin-left:4px}
.wind .big{color:var(--wind)}.solar .big{color:var(--solar)}
.track{height:10px;border-radius:6px;background:var(--field);border:1px solid var(--line);margin-top:12px;overflow:hidden}
.fill{height:100%;border-radius:6px;width:0;transition:width .35s ease}
.wind .fill{background:linear-gradient(90deg,#46c7dc,var(--wind))}.solar .fill{background:linear-gradient(90deg,#ffd166,var(--solar))}
.pct{font-size:12px;color:var(--mute);margin-top:6px}
.note{font-size:13px;color:var(--mute);margin:14px 0 0}.err{color:#b3261e;font-weight:600}@media(prefers-color-scheme:dark){.err{color:#ff8a80}}
.warn{font-size:13px;background:rgba(201,122,0,.12);border:1px solid var(--solar);border-radius:9px;padding:8px 10px;margin:12px 0 0}
</style></head><body><main>
<header>
<svg width="52" height="52" viewBox="0 0 52 52" aria-hidden="true"><circle cx="19" cy="19" r="8" fill="var(--solar)"/><g stroke="var(--solar)" stroke-width="2.4" stroke-linecap="round"><path d="M19 4v4M19 30v4M4 19h4M30 19h4M8.5 8.5l2.8 2.8M26.7 26.7l2.8 2.8M8.5 29.5l2.8-2.8M26.7 11.3l2.8-2.8"/></g><path d="M18 47c0-12 4-14 10-14h13a5 5 0 1 0-5-5" fill="none" stroke="var(--wind)" stroke-width="3" stroke-linecap="round"/><path d="M26 41h15a4 4 0 1 1-4 4" fill="none" stroke="var(--wind)" stroke-width="3" stroke-linecap="round"/></svg>
<h1>Solar and Wind energy prediction</h1></header>
<p class="lead">Choose a state and a date, then set the day's weather with the sliders or type exact values. The wind and solar energy update as you change them.</p>
<div class="grid">
<section class="card" aria-label="Weather inputs">
<div class="two"><div><label for="state">State</label><select id="state"></select></div><div><label for="date">Date</label><input id="date" type="date"></div></div>
<div class="wx" id="wx"></div>
<div class="btns"><button id="reset" type="button">Reset to typical values</button></div>
</section>
<section class="card res" aria-label="Estimated energy" aria-live="polite">
<h2>Estimated energy for the day</h2>
<div class="cond" id="cond"></div>
<div class="meter wind"><div class="mh"><svg width="40" height="40" viewBox="0 0 40 40" aria-hidden="true"><path d="M20 22v14" stroke="var(--wind)" stroke-width="3" stroke-linecap="round"/><g fill="var(--wind)"><path d="M20 20c-1-6-1-11 0-16 1 5 1 10 0 16z"/><path d="M20 20c5-3 9-6 14-6-3 4-8 7-14 6z"/><path d="M20 20c-5-3-9-6-14-6 3 4 8 7 14 6z" transform="rotate(0)"/></g><circle cx="20" cy="20" r="2.6" fill="var(--wind)"/></svg>
<div><b>Wind energy</b><span class="big" id="wv">–<small>GWh</small></span></div></div>
<div class="track"><div class="fill" id="wf"></div></div><div class="pct" id="wp"></div></div>
<div class="meter solar"><div class="mh"><svg width="40" height="40" viewBox="0 0 40 40" aria-hidden="true"><circle cx="20" cy="20" r="8" fill="var(--solar)"/><g stroke="var(--solar)" stroke-width="2.6" stroke-linecap="round"><path d="M20 4v5M20 31v5M4 20h5M31 20h5M8.7 8.7l3.5 3.5M27.8 27.8l3.5 3.5M8.7 31.3l3.5-3.5M27.8 12.2l3.5-3.5"/></g></svg>
<div><b>Solar energy</b><span class="big" id="sv">–<small>GWh</small></span></div></div>
<div class="track"><div class="fill" id="sf"></div></div><div class="pct" id="sp"></div></div>
<div id="warn"></div><p class="note" id="msg" role="status"></p>
</section>
</div>
</main>
<script>
const VARS=[["T2M","🌡️","Temperature","°C","0.1"],["RH2M","💧","Humidity","%","1"],["PRECIPITATION","🌧️","Rain","mm/day","0.1"],["WS10M","🍃","Wind speed at 10 m","m/s","0.1"],["WS50M","🌬️","Wind speed at 50 m","m/s","0.1"],["ALLSKY_SFC_SW_DWN","☀️","Sunlight","kWh/m²/day","0.1"],["CLOUD_AMT","☁️","Cloud cover","%","1"]];
const $=id=>document.getElementById(id),T=s=>s.replace(/\b\w/g,c=>c.toUpperCase());
const EXT={T2M:[-10,60]};   // slider limits wider than the data (temperature up to 60 °C)
let info,ticket=0,timer;
const num=k=>parseFloat($(k).value);
function conditions(){
  const c=[],t=num("T2M"),r=num("PRECIPITATION"),w=num("WS10M"),cl=num("CLOUD_AMT"),h=num("RH2M");
  if(r>=1)c.push("🌧️ Rainy");else if(cl>=70)c.push("☁️ Overcast");else if(cl>=30)c.push("⛅ Partly cloudy");else c.push("☀️ Clear sky");
  if(w>=7)c.push("💨 Windy");else if(w<2)c.push("🍃 Calm");else c.push("🍃 Breezy");
  c.push(t>=35?"🥵 Very hot":t>=28?"🌡️ Warm":t>=15?"🌡️ Mild":"🥶 Cool");
  if(h>=80)c.push("💧 Humid");
  $("cond").innerHTML=c.map(x=>`<span class="chip">${x}</span>`).join("");
}
function show(id,fid,pid,val,top){
  $(id).innerHTML=`${val.toFixed(2)}<small>GWh</small>`;
  const p=top>0?Math.min(100,100*val/top):0;$(fid).style.width=p+"%";
  $(pid).textContent=`${p.toFixed(0)}% of the highest day recorded for ${T($("state").value)} (${top.toFixed(2)})`;
}
async function calc(){
  conditions();const my=++ticket,weather={};VARS.forEach(([k])=>weather[k]=$(k).value);
  try{
    const r=await fetch("/api/estimate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({state:$("state").value,date:$("date").value,weather})});
    const j=await r.json();if(my!==ticket)return;
    if(!r.ok){$("msg").textContent=j.error;$("msg").className="note err";$("warn").innerHTML="";return}
    const top=info.top[$("state").value];
    show("wv","wf","wp",j.wind,top.wind);show("sv","sf","sp",j.solar,top.solar);
    $("warn").innerHTML=j.warnings.map(w=>`<div class="warn">⚠️ ${w}</div>`).join("");
    const m=info.weather_metrics;$("msg").className="note";
    $("msg").textContent=m?`Estimated by a weather-based model. Test accuracy: wind RMSE ${m.wind.rmse.toFixed(3)} (R² ${m.wind.r2.toFixed(3)}), solar RMSE ${m.solar.rmse.toFixed(3)} (R² ${m.solar.r2.toFixed(3)}).`:"";
  }catch(e){$("msg").textContent="Could not reach the server.";$("msg").className="note err"}
}
const later=()=>{clearTimeout(timer);timer=setTimeout(calc,200)};
function fill(){VARS.forEach(([k])=>{$(k).value=info.defaults[k];$(k+"_r").value=info.defaults[k]});calc()}
fetch("/api/info").then(r=>r.json()).then(j=>{
  info=j;j.states.forEach(s=>$("state").add(new Option(T(s),s)));
  $("date").value=new Date().toISOString().slice(0,10);
  $("wx").innerHTML=VARS.map(([k,ic,l,u,st])=>{const g=EXT[k]?{min:Math.min(j.ranges[k].min,EXT[k][0]),max:Math.max(j.ranges[k].max,EXT[k][1])}:j.ranges[k];return `<div class="row"><div class="top"><label for="${k}"><i>${ic}</i>${l}</label><span class="val"><input type="number" id="${k}" step="${st}" value="${j.defaults[k]}"><em>${u}</em></span></div><input type="range" id="${k}_r" aria-label="${l}" min="${g.min}" max="${g.max}" step="${st}" value="${j.defaults[k]}"><div class="rg"><span>${g.min.toFixed(1)}</span><span>${g.max.toFixed(1)}</span></div></div>`}).join("");
  VARS.forEach(([k])=>{
    $(k+"_r").oninput=e=>{$(k).value=e.target.value;later()};
    $(k).oninput=e=>{$(k+"_r").value=e.target.value;later()};
  });
  $("state").onchange=calc;$("date").onchange=calc;$("reset").onclick=fill;calc();
});
</script></body></html>"""


if __name__ == "__main__" and "ipykernel" not in sys.modules:       # not auto-run inside a Jupyter notebook
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="serve", choices=["train", "serve"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--port", type=int, default=5000)
    a, _ = ap.parse_known_args()
    (train if a.command == "train" else serve)(*((find_csv(a.csv), a.epochs) if a.command == "train" else (find_csv(a.csv), a.port)))