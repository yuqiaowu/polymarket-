from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_env_names(path: Path = ENV_PATH) -> Dict[str, bool]:
    """Return configured env variable names without exposing values."""
    if not path.exists():
        return {}

    names: Dict[str, bool] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        names[key] = bool(value.strip())
    return names


def load_env_values(path: Path = ENV_PATH) -> Dict[str, str]:
    """Return env values for runtime use. Do not include this in reports."""
    if not path.exists():
        return {}

    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            cleaned = value.strip()
            if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
                cleaned = cleaned[1:-1]
            values[key] = cleaned
    return values


def configured(keys: Iterable[str], env_names: Dict[str, bool]) -> Dict[str, bool]:
    return {key: bool(env_names.get(key)) for key in keys}


@dataclass(frozen=True)
class RuntimeConfig:
    env_path: Path = ENV_PATH
    reports_dir: Path = REPORTS_DIR
    request_timeout_seconds: int = 15


TRADE_SYMBOLS = ["TQQQ", "SQQQ", "SOXL", "SOXS"]
REFERENCE_SYMBOLS = ["QQQ", "SOXX", "SMH", "SPY"]
VOL_SYMBOLS = ["VIX", "VIX9D", "VIX3M"]
MACRO_SYMBOLS = VOL_SYMBOLS + ["US2Y", "US5Y", "US10Y", "DXY", "USDJPY"]

STOOQ_SYMBOLS = {
    "TQQQ": "tqqq.us",
    "SQQQ": "sqqq.us",
    "SOXL": "soxl.us",
    "SOXS": "soxs.us",
    "QQQ": "qqq.us",
    "SOXX": "soxx.us",
    "SMH": "smh.us",
    "SPY": "spy.us",
    "US2Y": "2yusy.b",
    "US5Y": "5yusy.b",
    "US10Y": "10yusy.b",
    "DXY": "dx.f",
    "USDJPY": "usdjpy",
}

CBOE_HISTORY_URLS = {
    "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VIX9D": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
}

POLYMARKET_THEME_SLUGS = {
    "AI_BUBBLE_RISK": ["ai-bubble-burst-by"],
    "AI_SEMICONDUCTOR_SENTIMENT": ["largest-company-end-of-december-2026"],
    "SEMICONDUCTOR_IPO_WINDOW": ["ipos-before-2027"],
    "CHINA_TAIWAN_RISK": ["will-china-invade-taiwan-by-june-30-2026"],
}

KALSHI_GPU_MARKETS = {
    "CURRENT_GEN_GPU_TIGHTNESS": [
        "KXB200W-26MAY01-4.8283",
    ],
    "LEGACY_GPU_PRICE_PRESSURE": [
        "KXH200W-26MAY01-3.5854",
        "KXH100W-26MAY01-1.8996",
        "KXA100W-26MAY01-1.0821",
    ],
    "CONSUMER_GPU_TIGHTNESS": [
        "KXRTX5090W-26MAY01-0.4567",
    ],
}

THEME_KEYWORDS = {
    "RATES_POLICY": ["fed", "rate", "interest", "fomc", "warsh"],
    "INFLATION_DATA": ["cpi", "pce", "ppi", "inflation"],
    "EMPLOYMENT_GROWTH": ["payroll", "unemployment", "jobs", "employment", "adp"],
    "AI_SEMICONDUCTOR": ["nvidia", "ai", "semiconductor", "cerebras", "chip", "gpu"],
    "AI_BUBBLE_RISK": ["ai bubble", "ai industry downturn"],
    "GPU_COMPUTE_SUPPLY": ["h100", "h200", "b200", "a100", "rtx 5090", "compute"],
    "WAR_GEOPOLITICS": ["war", "ceasefire", "iran", "ukraine", "gaza", "israel"],
    "CHINA_TAIWAN_RISK": ["china", "taiwan", "invade"],
    "TARIFF_TRADE_POLICY": ["tariff", "trade", "export control"],
    "US_ELECTION_POLICY": ["election", "president", "nominee"],
    "MARKET_VOLATILITY": ["vix", "volatility", "crash", "recession"],
}

DISCOVERY_DIRECTIONS = {
    "US_EQUITY_MARKET": {
        "description": "US stock-market direction, macro risk, rates, volatility, inflation, recession, and policy shocks.",
        "polymarket_keywords": [
            "fed",
            "interest rates",
            "rate cut",
            "cpi",
            "inflation",
            "pce",
            "recession",
            "nasdaq",
            "s&p",
            "stock market",
            "vix",
            "tariff",
            "ceasefire",
            "war",
        ],
        "kalshi_series": [
            "KXFEDDECISION",
            "KXCPI",
            "KXNASDAQ100",
            "KXGDP",
        ],
    },
    "AI_SEMICONDUCTOR": {
        "description": "AI infrastructure, semiconductor sentiment, GPU compute pricing, HBM/memory pressure, Taiwan risk, and AI bubble risk.",
        "polymarket_keywords": [
            "ai",
            "nvidia",
            "semiconductor",
            "chip",
            "gpu",
            "h100",
            "h200",
            "b200",
            "cerebras",
            "taiwan",
            "ai bubble",
            "ipo",
        ],
        "kalshi_series": [],
    },
}
