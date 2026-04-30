from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional


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
