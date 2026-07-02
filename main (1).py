#!/usr/bin/env python3
import os, json, argparse, math, random
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

OUT = "outputs"

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    down = -d.clip(upper=0).rolling(n).mean()
    rs = up/(down.replace(0,np.nan))
    return (100 - 100/(1+rs)).fillna(50)

def zscore(cs: pd.Series):
    return (cs - cs.mean())/(cs.std()+1e-9)

def sector_neutralize(values: pd.Series, sectors_vec: pd.Series):
    df = pd.DataFrame({"v":values, "sector":sectors_vec})
    mu = df.groupby("sector")["v"].transform("mean")
    return (values - mu).fillna(0.0)

def load_data_synthetic():
    np.random.seed(11)
    tickers = [f"STK{str(i).zfill(3)}" for i in range(1, 151)]
    sectors = ["Financials","IT","Energy","Materials","Consumer","Healthcare","Industrials","Utilities","Telecom","RealEstate"]
    sector_map: Dict[str,str] = {t: random.choice(sectors) for t in tickers}

    start = pd.Timestamp("2019-01-01")
    end   = pd.Timestamp("2025-08-01")
    dates = pd.bdate_range(start, end, freq="C")

    idx_mu, idx_sigma = 0.00035, 0.011
    idx_rets = np.random.normal(idx_mu, idx_sigma, len(dates))
    index_curve = 100*np.exp(np.cumsum(idx_rets))
    index = pd.Series(index_curve, index=dates, name="NIFTY")

    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    volumes = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in tickers:
        mu = idx_mu + np.random.normal(0, 0.00015)
        sigma = idx_sigma*np.random.uniform(0.8, 1.4)
        eps = np.random.normal(mu, sigma, len(dates))
        prices[t] = 100*np.exp(np.cumsum(eps + np.random.normal(0, 0.0004, len(dates))))
        base_vol = np.random.randint(8e5, 8e6)
        volumes[t] = base_vol * (1 + 0.25*np.random.randn(len(dates))).clip(1e5, None)

    fund_q = pd.date_range(start, end, freq="Q")
    fund_rows = []
    for t in tickers:
        roe = np.clip(np.random.normal(0.16, 0.05, len(fund_q)), 0.02, 0.4)
        de  = np.clip(np.random.normal(0.6,  0.35, len(fund_q)), 0.0, 3.0)
        pm  = np.clip(np.random.normal(0.12, 0.07, len(fund_q)), -0.05, 0.4)
        fund_rows.append(pd.DataFrame({"date":fund_q, "ticker":t, "ROE":roe, "DE":de, "PM":pm}))
    fund = pd.concat(fund_rows, ignore_index=True)
    fund_panel = fund.pivot(index="date", columns="ticker", values=["ROE","DE","PM"]).sort_index()
    fund_panel = fund_panel.reindex(dates, method="ffill").fillna(method="bfill")
    return prices, volumes, fund_panel, index, sectors, sector_map

def load_data_csv(data_dir):
    prices = pd.read_csv(os.path.join(data_dir, "prices.csv"), parse_dates=["date"]).set_index("date")
    volumes = pd.read_csv(os.path.join(data_dir, "volumes.csv"), parse_dates=["date"]).set_index("date")
    fundamentals = pd.read_csv(os.path.join(data_dir, "fundamentals.csv"), parse_dates=["date"])
    sector_map_df = pd.read_csv(os.path.join(data_dir, "sectors.csv"))
    sector_map = dict(zip(sector_map_df["ticker"], sector_map_df["sector"]))
    sectors = sorted(sector_map_df["sector"].unique().tolist())

    dates = prices.index
    index = (prices.mean(axis=1).fillna(method="ffill")).rename("NIFTY")  # replace with official index if available
    fund_panel = fundamentals.pivot(index="date", columns="ticker", values=["ROE","DE","PM"]).sort_index()
    fund_panel = fund_panel.reindex(dates, method="ffill").fillna(method="bfill")
    return prices, volumes, fund_panel, index, sectors, sector_map

def run_pipeline(prices, volumes, fund_panel, index, sectors, sector_map,
                 top_n=35, sector_cap=0.25, name_cap=0.07, target_vol=0.15, cost_bps=12, horizon=20):
    # Factors
    mom_6m = prices.apply(lambda s: s/s.shift(126) - 1.0)
    vol_6m = prices.pct_change().rolling(126).std()
    rsi_14 = prices.apply(lambda s: rsi(s, 14))
    adv_1m = volumes.rolling(22).mean()
    liq_mask = adv_1m > 4e5

    # Rebalance months
    cal = pd.DataFrame({"date": prices.index})
    cal["month"] = cal["date"].dt.to_period("M")
    rebal_dates = cal.groupby("month")["date"].max().tolist()

    # Panel
    panel_rows = []
    for d in rebal_dates[:-2]:
        if d not in prices.index: 
            continue
        px = prices.loc[d]
        ok = liq_mask.loc[d] & px.notna()
        names = px.index[ok]
        if len(names) < 40: 
            continue
        feats = pd.DataFrame(index=names)
        feats["date"] = d
        feats["ticker"] = names
        feats["sector"] = [sector_map.get(t, "Unknown") for t in names]
        feats["mom6"] = zscore(mom_6m.loc[d, names])
        feats["vol6"] = -zscore(vol_6m.loc[d, names])  # lower vol better
        feats["rsi14"] = zscore(rsi_14.loc[d, names])
        feats["roe"] = zscore(fund_panel["ROE"].loc[d, names])
        feats["de"]  = -zscore(fund_panel["DE"].loc[d, names])   # lower better
        feats["pm"]  = zscore(fund_panel["PM"].loc[d, names])
        for k in ["mom6","roe","pm"]:
            feats[k+"_sn"] = sector_neutralize(feats[k], feats["sector"])

        loc0 = prices.index.get_loc(d)
        if loc0+horizon >= len(prices.index): 
            continue
        d1 = prices.index[loc0 + horizon]
        r_stock = (prices.loc[d1, names]-prices.loc[d, names])/prices.loc[d, names]
        r_idx   = (index.loc[d1] - index.loc[d]) / index.loc[d]
        label_ex = r_stock - float(r_idx)
        feats["label_up"] = (label_ex > 0).astype(int)
        feats["label_excess"] = label_ex.values
        panel_rows.append(feats.reset_index(drop=True))
    panel = pd.concat(panel_rows, ignore_index=True)

    # ML
    feature_cols = ["mom6_sn","roe_sn","pm_sn","vol6","rsi14","de"]
    panel = panel.dropna(subset=feature_cols+["label_up"])
    split = panel["date"].quantile(0.75)
    train = panel[panel["date"] <= split]
    test  = panel[panel["date"] >  split]

    if SKLEARN_OK and len(train) > 1000:
        rf = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=10, random_state=7, n_jobs=-1)
        rf.fit(train[feature_cols], train["label_up"])
        test = test.copy()
        test["ml_proba"] = rf.predict_proba(test[feature_cols])[:,1]
        auc = float(roc_auc_score(test["label_up"], test["ml_proba"]))
        feat_imp = dict(zip(feature_cols, rf.feature_importances_))
    else:
        rf = None
        test = test.copy()
        test["ml_proba"] = (test["mom6_sn"] + test["roe_sn"] + test["pm_sn"] - test["de"]) / 4.0
        g = test.groupby("date")["ml_proba"]
        test["ml_proba"] = (test["ml_proba"] - g.transform("min")) / (g.transform("max") - g.transform("min") + 1e-9)
        try:
            auc = float(roc_auc_score(test["label_up"], test["ml_proba"]))
        except Exception:
            auc = float("nan")
        feat_imp = {c: np.nan for c in feature_cols}

    def composite(df):
        quality = zscore(df["roe_sn"] + df["pm_sn"] - df["de"])
        score = 0.4*zscore(df["mom6_sn"]) + 0.3*quality + 0.3*zscore(df["ml_proba"])
        return score

    test["score"] = test.groupby("date", group_keys=False).apply(composite)

    # Portfolio
    def pick_port(group):
        g = group.sort_values("score", ascending=False).copy()
        max_per_sector = max(1, int(top_n*sector_cap))
        picks = []
        sector_counts = {s:0 for s in sectors}
        for _, row in g.iterrows():
            s = row["sector"]
            if sector_counts.get(s,0) < max_per_sector and len(picks) < top_n:
                picks.append(row); sector_counts[s]=sector_counts.get(s,0)+1
            if len(picks) >= top_n: break
        port = pd.DataFrame(picks).reset_index(drop=True)
        port["raw_w"] = min(1.0/top_n, name_cap)
        port["weight"] = port["raw_w"] / port["raw_w"].sum()
        return port[["date","ticker","sector","weight"]]

    # pull params from closure
    top_n, sector_cap, name_cap = 35, 0.25, 0.07  # default; can be overridden in main()
    ports = test.groupby("date", group_keys=False).apply(pick_port).reset_index(drop=True)

    # Beta hedge & backtest
    rets = prices.pct_change().fillna(0.0)
    idxr = index.pct_change().fillna(0.0)
    roll = 60
    betas = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for t in prices.columns:
        x = idxr.rolling(roll).var()
        cov = rets[t].rolling(roll).cov(idxr)
        betas[t] = (cov / (x + 1e-9)).fillna(1.0)

    bt_rows = []
    w_prev = {}
    TARGET_VOL = target_vol
    COST = cost_bps/10000.0

    for d, grp in ports.groupby("date"):
        loc0 = prices.index.get_loc(d)
        d1 = prices.index[min(loc0+horizon, len(prices.index)-1)]
        r_vec = (prices.loc[d1, grp.ticker].values - prices.loc[d, grp.ticker].values)/prices.loc[d, grp.ticker].values
        beta_p = float(np.nansum(grp["weight"].values * betas.loc[d, grp.ticker].values))
        idx_ret = float((index.loc[d1]-index.loc[d]) / index.loc[d])
        port_ret = float(np.dot(grp["weight"].values, r_vec))
        hedged_ret = port_ret - beta_p*idx_ret

        # vol target
        if bt_rows:
            past = pd.Series([row["hedged_ret"] for row in bt_rows[-12:]])
            ann_vol = past.std()*np.sqrt(12) if len(past)>2 else idxr.rolling(252).std().reindex(prices.index).ffill().loc[d]*np.sqrt(252)
        else:
            ann_vol = idxr.rolling(252).std().reindex(prices.index).ffill().loc[d]*np.sqrt(252)
        scale = float(np.clip(TARGET_VOL/(ann_vol+1e-9), 0.25, 2.0))

        # costs
        w_now = {t: float(w) for t, w in zip(grp["ticker"], grp["weight"]) }
        all_names = set(w_prev) | set(w_now)
        turnover = sum(abs(w_now.get(n,0)-w_prev.get(n,0)) for n in all_names)
        cost = turnover * COST

        ret_scaled = scale*hedged_ret - cost
        bt_rows.append({"date_exit": d1, "ret_net": ret_scaled, "idx_ret": idx_ret, "hedged_ret": hedged_ret, "turnover": turnover, "scale": scale})
        w_prev = w_now

    bt = pd.DataFrame(bt_rows).sort_values("date_exit").reset_index(drop=True)
    bt["curve_net"] = (1 + bt["ret_net"]).cumprod()
    bt["curve_idx"] = (1 + bt["idx_ret"]).cumprod()

    def cagr(curve, periods=12):
        if len(curve)<2: return np.nan
        total = curve.iloc[-1]/curve.iloc[0]
        years = len(curve)/periods
        return total**(1/years)-1

    def max_dd(curve):
        rollm = curve.cummax()
        return (curve/rollm - 1).min()

    def sharpe(returns, periods=12):
        mu = returns.mean()*periods
        sd = returns.std()*np.sqrt(periods)+1e-9
        return mu/sd

    metrics = {
        "AUC(test)": None if ('auc' not in locals() or (isinstance(auc,float) and np.isnan(auc))) else round(float(auc),3),
        "CAGR(Net)": round(float(cagr(bt["curve_net"])),4),
        "CAGR(Index)": round(float(cagr(bt["curve_idx"])),4),
        "Sharpe(Net)": round(float(sharpe(bt["ret_net"])),3),
        "MaxDD(Net)": round(float(max_dd(bt["curve_net"])),4),
        "Avg Turnover": round(float(bt["turnover"].mean()),3),
        "Avg Scale": round(float(bt["scale"].mean()),3),
    }

    os.makedirs(OUT, exist_ok=True)
    bt.to_csv(os.path.join(OUT, "backtest_net.csv"), index=False)
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Charts
    plt.figure(figsize=(9,5))
    plt.plot(bt["date_exit"], bt["curve_net"], label="Strategy (net)")
    plt.plot(bt["date_exit"], bt["curve_idx"], label="Index")
    plt.legend(); plt.title("Equity Curve")
    plt.xlabel("Date"); plt.ylabel("Growth of 1")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "equity_curve.png")); plt.close()

    roll = 12
    excess_roll = (1 + (bt["ret_net"] - bt["idx_ret"])).rolling(roll).apply(np.prod, raw=True) - 1
    plt.figure(figsize=(9,5))
    plt.plot(bt["date_exit"], excess_roll)
    plt.title("Rolling 12M Excess Return (Net)")
    plt.xlabel("Date"); plt.ylabel("Excess")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "rolling_excess_12m.png")); plt.close()

    plt.figure(figsize=(9,5))
    plt.plot(bt["date_exit"], bt["turnover"])
    plt.title("Turnover per Rebalance"); plt.xlabel("Date"); plt.ylabel("Turnover")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "turnover.png")); plt.close()

    print(json.dumps(metrics, indent=2))

def main():
    ap = argparse.ArgumentParser(description="Momentum+Quality+ML Quant Strategy")
    ap.add_argument("--data-dir", type=str, default="", help="Directory with prices.csv, volumes.csv, fundamentals.csv, sectors.csv")
    ap.add_argument("--top-n", type=int, default=35)
    ap.add_argument("--sector-cap", type=float, default=0.25)
    ap.add_argument("--name-cap", type=float, default=0.07)
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--cost-bps", type=float, default=12.0)
    ap.add_argument("--horizon", type=int, default=20)
    args = ap.parse_args()

    if args.data_dir and os.path.exists(args.data_dir):
        prices, volumes, fund_panel, index, sectors, sector_map = load_data_csv(args.data_dir)
    else:
        prices, volumes, fund_panel, index, sectors, sector_map = load_data_synthetic()

    # Use CLI params by rebinding defaults inside run_pipeline via closure variables:
    global top_n, sector_cap, name_cap, target_vol, cost_bps, horizon
    top_n, sector_cap, name_cap = args.top_n, args.sector_cap, args.name_cap
    target_vol, cost_bps, horizon = args.target_vol, args.cost_bps, args.horizon

    run_pipeline(
        prices, volumes, fund_panel, index, sectors, sector_map,
        top_n=top_n, sector_cap=sector_cap, name_cap=name_cap,
        target_vol=target_vol, cost_bps=cost_bps, horizon=horizon
    )

if __name__ == "__main__":
    main()


