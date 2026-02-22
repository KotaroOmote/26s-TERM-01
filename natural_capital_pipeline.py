#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class PeriodConfig:
    start_date: str
    end_date: str


@dataclass
class DataConfig:
    cache_dir: str = "./cache"
    outputs_subdir: str = "fujisawa_demo"


@dataclass
class ModelConfig:
    lookback_weeks: int = 12
    garch_forecast_horizon: int = 8


@dataclass
class FinanceConfig:
    derivative_notional: float = 1_000_000.0
    derivative_vol_quantile: float = 0.9
    derivative_max_payout_ratio: float = 0.2
    bond_base_coupon: float = 0.045
    bond_step_down_bps: float = 30.0
    bond_step_up_bps: float = 40.0
    bond_target_quarterly_growth: float = 0.01


@dataclass
class PipelineConfig:
    period: PeriodConfig
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    finance: FinanceConfig = field(default_factory=FinanceConfig)


def load_config(path: Path) -> PipelineConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PipelineConfig(
        period=PeriodConfig(**raw["period"]),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        finance=FinanceConfig(**raw.get("finance", {})),
    )


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def date_to_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_range(start: date, end: date) -> List[date]:
    cur = week_start(start)
    out = []
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=7)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, headers: List[str], rows: List[List[Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def synthetic_weekly_rows(cfg: PipelineConfig) -> List[Dict[str, Any]]:
    weeks = week_range(parse_date(cfg.period.start_date), parse_date(cfg.period.end_date))
    n = max(1, len(weeks))
    rows = []
    for i, w in enumerate(weeks):
        t = i / max(1, n - 1)
        rows.append(
            {
                "week_start": date_to_str(w),
                "s2_green_fraction": 0.45 + 0.2 * t,
                "s2_fragmentation": 0.30 - 0.12 * t,
                "gbif_species_richness": 10 + 2.5 * np.sin(i / 4.0),
                "weather_soil_moisture": 0.24 + 0.05 * np.sin(i / 8.0),
                "weather_temp_stress": 0.45 + 0.08 * np.sin(i / 9.0),
            }
        )
    return rows


def build_natural_capital_index(
    weekly_rows: List[Dict[str, Any]], lookback_weeks: int = 12
) -> Dict[str, Any]:
    del lookback_weeks

    if not weekly_rows:
        return {"rows": [], "feature_names": [], "weights": []}

    feature_names = [k for k in weekly_rows[0].keys() if k != "week_start"]
    x = np.array([[float(r[k]) for k in feature_names] for r in weekly_rows], dtype=np.float64)

    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-9] = 1.0
    z = (x - mu) / sd

    score = z.mean(axis=1)

    ret = np.zeros(len(score), dtype=np.float64)
    if len(score) > 1:
        ret[1:] = np.clip(0.08 * np.diff(score), -0.3, 0.3)

    idx = np.zeros(len(score), dtype=np.float64)
    idx[0] = 100.0
    for i in range(1, len(idx)):
        idx[i] = max(1e-6, idx[i - 1] * (1.0 + ret[i]))

    out_rows = []
    for i, row in enumerate(weekly_rows):
        row_out = dict(row)
        row_out["natural_capital_index"] = float(idx[i])
        row_out["ecological_return"] = float(ret[i])
        out_rows.append(row_out)

    weights = [1.0 / len(feature_names)] * len(feature_names)
    return {"rows": out_rows, "feature_names": feature_names, "weights": weights}


class Garch11:
    def __init__(self, alpha: float = 0.08, beta: float = 0.9):
        self.alpha = alpha
        self.beta = beta
        self.omega = 0.0
        self.mu = 0.0
        self.sigma2 = np.array([], dtype=np.float64)

    def fit(self, returns: np.ndarray) -> "Garch11":
        r = np.asarray(returns, dtype=np.float64)
        self.mu = float(np.mean(r))
        e = r - self.mu
        var = float(np.var(e)) + 1e-8

        if self.alpha + self.beta >= 0.995:
            self.beta = 0.995 - self.alpha

        self.omega = max(1e-10, var * (1.0 - self.alpha - self.beta))

        s2 = np.zeros(len(r), dtype=np.float64)
        s2[0] = var
        for t in range(1, len(r)):
            s2[t] = self.omega + self.alpha * (e[t - 1] ** 2) + self.beta * s2[t - 1]
            s2[t] = max(1e-10, s2[t])
        self.sigma2 = s2
        return self

    @staticmethod
    def forecast(model: "Garch11", horizon: int) -> np.ndarray:
        h = max(1, int(horizon))
        if len(model.sigma2) == 0:
            return np.full(h, 1e-6, dtype=np.float64)

        p = min(0.999, model.alpha + model.beta)
        long_run = model.omega / max(1e-8, 1.0 - p)

        out = np.zeros(h, dtype=np.float64)
        prev = float(model.sigma2[-1])
        for i in range(h):
            prev = long_run + p * (prev - long_run)
            out[i] = max(1e-10, prev)
        return out


def build_finance_signals(
    cfg: PipelineConfig, vol_hist: np.ndarray, vol_fore: np.ndarray
) -> Dict[str, Any]:
    threshold = float(np.quantile(vol_hist, cfg.finance.derivative_vol_quantile))
    triggered = bool(np.max(vol_fore) > threshold)

    excess = max(0.0, float(np.max(vol_fore) / max(1e-10, threshold) - 1.0))
    payout_ratio = min(cfg.finance.derivative_max_payout_ratio, 0.5 * excess)
    payout = payout_ratio * cfg.finance.derivative_notional

    coupon = cfg.finance.bond_base_coupon
    recent = float(np.mean(vol_hist[-13:])) if len(vol_hist) >= 13 else float(np.mean(vol_hist))
    prev = float(np.mean(vol_hist[-26:-13])) if len(vol_hist) >= 26 else recent
    growth_proxy = (prev - recent) / prev if prev > 1e-10 else 0.0
    if growth_proxy >= cfg.finance.bond_target_quarterly_growth:
        coupon -= cfg.finance.bond_step_down_bps / 10000.0
    else:
        coupon += cfg.finance.bond_step_up_bps / 10000.0

    return {
        "derivative": {
            "vol_threshold": threshold,
            "triggered": triggered,
            "payout_ratio": payout_ratio,
            "payout_amount": payout,
        },
        "nature_bond": {
            "base_coupon": cfg.finance.bond_base_coupon,
            "final_coupon": coupon,
            "quarterly_growth_proxy": growth_proxy,
        },
    }


def build_data_cache(cfg: PipelineConfig, out_root: Path) -> Dict[str, Any]:
    raw = out_root / "raw"
    ensure_dir(raw)
    weekly_rows = synthetic_weekly_rows(cfg)

    write_json(raw / "sentinel2_items.json", {"features": []})
    write_json(raw / "sentinel1_items.json", {"features": []})
    write_json(raw / "sentinel2_thumbnails.json", [])
    write_json(raw / "sentinel1_thumbnails.json", [])
    write_json(raw / "gbif_occurrences.json", {"results": []})
    write_json(raw / "gbif_images.json", [])
    write_json(raw / "weather_daily.json", {"daily": {}})
    write_json(raw / "weekly_rows.json", weekly_rows)
    return {"weekly_rows": weekly_rows}


def load_cached_data(out_root: Path) -> Dict[str, Any]:
    path = out_root / "raw" / "weekly_rows.json"
    return {"weekly_rows": read_json(path) if path.exists() else []}


def run_modeling_from_data(
    cfg: PipelineConfig, out_root: Path, data_bundle: Dict[str, Any]
) -> Dict[str, Any]:
    weekly_rows = data_bundle.get("weekly_rows", []) or synthetic_weekly_rows(cfg)

    idx_pack = build_natural_capital_index(weekly_rows, cfg.model.lookback_weeks)
    idx_rows = idx_pack["rows"]

    returns = np.array([r["ecological_return"] for r in idx_rows], dtype=np.float64)
    garch = Garch11().fit(returns)
    garch_fore = Garch11.forecast(garch, cfg.model.garch_forecast_horizon)

    tf_like_fore = np.full(
        cfg.model.garch_forecast_horizon, float(np.mean(garch.sigma2[-4:])), dtype=np.float64
    )

    signals = build_finance_signals(cfg, garch.sigma2, tf_like_fore)

    derived = out_root / "derived"
    ensure_dir(derived)

    headers = list(weekly_rows[0].keys())
    write_csv(derived / "features_weekly.csv", headers, [[r[k] for k in headers] for r in weekly_rows])

    write_csv(
        derived / "natural_capital_index.csv",
        ["week_start", "natural_capital_index", "ecological_return"],
        [[r["week_start"], r["natural_capital_index"], r["ecological_return"]] for r in idx_rows],
    )

    write_csv(
        derived / "volatility_forecast.csv",
        ["horizon_week", "garch_sigma2", "transformer_sigma2"],
        [[i + 1, float(garch_fore[i]), float(tf_like_fore[i])] for i in range(cfg.model.garch_forecast_horizon)],
    )

    write_json(derived / "finance_signals.json", signals)
    write_json(
        derived / "model_meta.json",
        {"garch": {"mu": garch.mu, "omega": garch.omega, "alpha": garch.alpha, "beta": garch.beta}},
    )

    summary = {
        "n_weeks": len(idx_rows),
        "last_index": float(idx_rows[-1]["natural_capital_index"]),
        "last_return": float(idx_rows[-1]["ecological_return"]),
        "last_garch_sigma2": float(garch.sigma2[-1]),
        "max_forecast_sigma2": float(np.max(tf_like_fore)),
    }
    write_json(derived / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "fetch", "from-cache"])
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    out_root = Path(cfg.data.cache_dir) / cfg.data.outputs_subdir

    if args.command == "fetch":
        build_data_cache(cfg, out_root)
        print(f"[ok] cached raw data at: {out_root / 'raw'}")
        return

    if args.command == "from-cache":
        data = load_cached_data(out_root)
        if not data["weekly_rows"]:
            data = build_data_cache(cfg, out_root)
    else:
        data = build_data_cache(cfg, out_root)

    summary = run_modeling_from_data(cfg, out_root, data)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"[ok] outputs at: {out_root}")


if __name__ == "__main__":
    main()
