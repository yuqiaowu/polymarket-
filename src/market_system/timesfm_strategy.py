from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional


TIMESFM_SCHEMA_VERSION = "timesfm_forecast_v0.1"
TIMESFM_TRADE_SYMBOLS = ["TQQQ", "SQQQ", "SOXL", "SOXS"]


@dataclass
class TimesFMForecast:
    point_return_pct: float
    q10_return_pct: Optional[float]
    q50_return_pct: Optional[float]
    q90_return_pct: Optional[float]
    probability_positive: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


class TimesFMUnavailable(RuntimeError):
    pass


class TimesFMForecaster:
    def __init__(self, model_id: str, max_context: int, max_horizon: int) -> None:
        self.model_id = model_id
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.model = self._load_model(model_id, max_context, max_horizon)

    def forecast(self, closes: list[float], horizon: int) -> TimesFMForecast:
        import numpy as np

        if len(closes) < 2:
            raise ValueError("at least two closes are required")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon {horizon} exceeds max_horizon {self.max_horizon}")

        safe_closes = np.asarray([max(float(value), 1e-9) for value in closes], dtype=np.float32)
        log_closes = np.log(safe_closes)
        last_log = float(log_closes[-1])
        point_forecast, quantile_forecast = self.model.forecast(horizon=horizon, inputs=[log_closes])

        point_value = float(point_forecast[0][horizon - 1])
        point_return = _log_forecast_to_return_pct(point_value, last_log)

        q10 = q50 = q90 = probability_positive = None
        if quantile_forecast is not None:
            q_values = _quantile_values(quantile_forecast, horizon)
            if q_values:
                q10 = _log_forecast_to_return_pct(q_values.get("q10"), last_log)
                q50 = _log_forecast_to_return_pct(q_values.get("q50"), last_log)
                q90 = _log_forecast_to_return_pct(q_values.get("q90"), last_log)
                probability_positive = _probability_positive(q_values, last_log)

        if probability_positive is None:
            probability_positive = 1.0 if point_return > 0 else 0.0
        return TimesFMForecast(
            point_return_pct=round(point_return, 4),
            q10_return_pct=_round_optional(q10),
            q50_return_pct=_round_optional(q50),
            q90_return_pct=_round_optional(q90),
            probability_positive=_round_optional(probability_positive),
        )

    def _load_model(self, model_id: str, max_context: int, max_horizon: int) -> Any:
        try:
            import torch
            import timesfm
        except Exception as exc:  # noqa: BLE001
            raise TimesFMUnavailable(
                "TimesFM dependencies are not installed. Run: "
                "python -m pip install --index-url https://pypi.org/simple -r requirements-timesfm.txt"
            ) from exc

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        try:
            model_cls = getattr(timesfm, "TimesFM_2p5_200M_torch")
            model = model_cls.from_pretrained(model_id)
            model.compile(
                timesfm.ForecastConfig(
                    max_context=max_context,
                    max_horizon=max_horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            return model
        except Exception as exc:  # noqa: BLE001
            raise TimesFMUnavailable(f"Failed to load TimesFM model {model_id}: {exc}") from exc


def build_timesfm_forecast_matrix(
    forecasts: dict[str, TimesFMForecast],
    horizon_days: int,
    q10_floor: float = -10.0,
) -> dict:
    symbols = {}
    warnings = []
    for symbol in TIMESFM_TRADE_SYMBOLS:
        forecast = forecasts.get(symbol)
        if forecast is None:
            warnings.append(f"MISSING_TIMESFM_FORECAST_{symbol}")
            continue
        score = forecast_score(forecast, q10_floor=q10_floor)
        payload = forecast.to_dict()
        payload["forecast_score"] = score
        payload["candidate_hint"] = candidate_hint(forecast, q10_floor=q10_floor)
        symbols[symbol] = payload

    ranking = sorted(symbols.keys(), key=lambda item: symbols[item].get("forecast_score", -999.0), reverse=True)
    conflicts = _forecast_conflicts(symbols)
    return {
        "schema_version": TIMESFM_SCHEMA_VERSION,
        "strategy": "timesfm_forecast",
        "horizon": f"{horizon_days}D",
        "symbols": symbols,
        "ranking": ranking,
        "warnings": warnings + conflicts,
        "policy": (
            "TimesFM scores all four allowed leveraged ETF symbols. It ranks and filters candidates, "
            "but does not grant trade permission by itself."
        ),
    }


def forecast_score(forecast: TimesFMForecast, q10_floor: float = -10.0) -> float:
    q50 = forecast.q50_return_pct if forecast.q50_return_pct is not None else forecast.point_return_pct
    probability = forecast.probability_positive if forecast.probability_positive is not None else 0.5
    q10 = forecast.q10_return_pct
    normalized_q50 = max(-1.0, min(1.0, q50 / 5.0))
    probability_edge = max(-1.0, min(1.0, (probability - 0.5) * 4.0))
    if q10 is None:
        downside_adjustment = 0.0
    elif q10 >= q10_floor:
        downside_adjustment = min(1.0, (q10 - q10_floor) / abs(q10_floor or -1.0))
    else:
        downside_adjustment = -min(1.0, (q10_floor - q10) / abs(q10_floor or -1.0))
    score = 0.45 * normalized_q50 + 0.35 * probability_edge + 0.20 * downside_adjustment
    return round(max(-1.0, min(1.0, score)), 4)


def candidate_hint(forecast: TimesFMForecast, q10_floor: float = -10.0) -> str:
    q50 = forecast.q50_return_pct if forecast.q50_return_pct is not None else forecast.point_return_pct
    probability = forecast.probability_positive if forecast.probability_positive is not None else 0.5
    q10 = forecast.q10_return_pct
    if probability >= 0.55 and q50 > 0 and (q10 is None or q10 >= q10_floor):
        return "SUPPORT"
    if probability >= 0.50 and q50 >= 0:
        return "WEAK_SUPPORT"
    if probability < 0.45 or q50 < 0:
        return "OPPOSE"
    return "NEUTRAL"


def _forecast_conflicts(symbols: dict[str, dict]) -> list[str]:
    warnings = []
    if _is_support(symbols.get("TQQQ")) and _is_support(symbols.get("SQQQ")):
        warnings.append("FORECAST_CONFLICT_TQQQ_SQQQ")
    if _is_support(symbols.get("SOXL")) and _is_support(symbols.get("SOXS")):
        warnings.append("FORECAST_CONFLICT_SOXL_SOXS")
    return warnings


def _is_support(payload: Optional[dict]) -> bool:
    return bool(payload and payload.get("candidate_hint") in {"SUPPORT", "WEAK_SUPPORT"})


def _quantile_values(quantile_forecast: Any, horizon: int) -> dict[str, float]:
    values = quantile_forecast[0][horizon - 1]
    length = len(values)
    if length >= 10:
        return {"q10": float(values[1]), "q50": float(values[5]), "q90": float(values[9])}
    if length >= 9:
        return {"q10": float(values[0]), "q50": float(values[4]), "q90": float(values[8])}
    return {}


def _probability_positive(q_values: dict[str, float], last_log: float) -> Optional[float]:
    quantiles = [(0.10, q_values.get("q10")), (0.50, q_values.get("q50")), (0.90, q_values.get("q90"))]
    clean = [(prob, value) for prob, value in quantiles if value is not None]
    if not clean:
        return None
    threshold = last_log
    if clean[0][1] >= threshold:
        return 0.9
    if clean[-1][1] <= threshold:
        return 0.1
    for (left_p, left_v), (right_p, right_v) in zip(clean, clean[1:]):
        if left_v <= threshold <= right_v and right_v != left_v:
            cdf = left_p + (right_p - left_p) * ((threshold - left_v) / (right_v - left_v))
            return max(0.0, min(1.0, 1.0 - cdf))
    return None


def _log_forecast_to_return_pct(value: Optional[float], last_log: float) -> float:
    if value is None:
        return 0.0
    return (math.exp(float(value) - last_log) - 1.0) * 100.0


def _round_optional(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)
