from __future__ import annotations
from dataclasses import dataclass

from market_system.timesfm_strategy import TimesFMForecast


@dataclass
class VixSqueezeSignal:
    vix_current: float
    vix_forecast_mean: float
    vix_forecast_slope: float
    vix_forecast_peak_detected: bool
    action: str
    confidence: float
    target_exposure: float = 0.0
    reason: str = ""


class VixSqueezeEngine:
    """
    VIX Squeeze Strategy Engine.
    Uses VIX mean reversion patterns (Squeezes) to time TQQQ entries.
    """
    def __init__(self, high_vix_threshold: float = 25.0, low_vix_threshold: float = 15.0):
        self.high_vix = high_vix_threshold
        self.low_vix = low_vix_threshold

    def analyze_timesfm(self, vix_history: list[float], forecast: TimesFMForecast) -> VixSqueezeSignal:
        if not vix_history or not forecast:
            return VixSqueezeSignal(0, 0, 0, False, "NO_TRADE", 0.0)

        curr_vix = vix_history[-1]

        # 1. Calculate Momentum for Spike Protection.
        # Daily VIX points per day over the last three closes.
        short_window = vix_history[-3:]
        if len(short_window) >= 3:
            slope_3d = (short_window[-1] - short_window[0]) / 2.0
        else:
            slope_3d = 0.0

        # 2. Mean Reversion Point (The Peak)
        prob_pos = forecast.probability_positive if forecast.probability_positive is not None else 0.5
        q50 = forecast.q50_return_pct if forecast.q50_return_pct is not None else forecast.point_return_pct

        is_peaking = (prob_pos < 0.45) and (q50 < -2.0)

        # 3. Detect Spike Risk (Complacency)
        is_spiking = (prob_pos > 0.55) and (q50 > 2.0)

        action = "NO_TRADE"
        confidence = 0.0
        target_exposure = 0.0
        reason = "NO_SIGNAL"

        # --- AI-CONFIRMED VIX SQUEEZE STRATEGY ---
        # 1. Spike Protection: if VIX slope suddenly jumps, reduce exposure unless
        # TimesFM strongly disagrees and sees VIX rolling over.
        if slope_3d > 1.5:
            action = "HEDGE"
            confidence = 0.95
            reason = "SPIKE_FULL_HEDGE"
            target_exposure = 0.0
            if q50 < -20.0 or prob_pos < 0.25:
                action = "REDUCE_RISK"
                target_exposure = 0.75
                confidence = 0.75
                reason = "SPIKE_TIMESFM_VERY_STRONG_REVERSAL"
            elif q50 < -12.0 or prob_pos < 0.35:
                action = "REDUCE_RISK"
                target_exposure = 0.50
                confidence = 0.80
                reason = "SPIKE_TIMESFM_STRONG_REVERSAL"
            elif q50 < -8.0 or prob_pos < 0.40:
                action = "REDUCE_RISK"
                target_exposure = 0.25
                confidence = 0.85
                reason = "SPIKE_TIMESFM_MODERATE_REVERSAL"

        # 2. Squeeze Buy: extreme fear, but AI sees VIX topping and rolling over.
        elif curr_vix > self.high_vix:
            if is_peaking:
                action = "LONG_TQQQ"
                confidence = 0.95
                target_exposure = 1.0
                reason = "SQUEEZE_BUY"
            else:
                action = "NO_CHANGE"
                target_exposure = -1.0
                reason = "HIGH_VIX_NO_AI_CONFIRMATION"

        # 3. Complacency Exit: low VIX, but AI sees VIX bottoming and rising.
        elif curr_vix < self.low_vix:
            if is_spiking:
                action = "HEDGE"
                confidence = 0.9
                target_exposure = 0.0
                reason = "COMPLACENCY_EXIT"
            else:
                action = "NO_CHANGE"
                target_exposure = -1.0
                reason = "LOW_VIX_NO_AI_CONFIRMATION"

        # 4. ZONE: NOISE (15 <= VIX <= 25) -> Maintain current state
        else:
            action = "NO_CHANGE"
            confidence = 0.5
            target_exposure = -1.0
            reason = "NOISE_MAINTAIN"

        return VixSqueezeSignal(
            vix_current=curr_vix,
            vix_forecast_mean=q50,
            vix_forecast_slope=slope_3d,
            vix_forecast_peak_detected=is_peaking,
            action=action,
            confidence=confidence,
            target_exposure=target_exposure,
            reason=reason,
        )
