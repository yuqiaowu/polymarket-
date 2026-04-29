from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from .qlib_strategy import QlibForecaster

@dataclass
class QlibVixSignal:
    vix_current: float
    action: str  # LONG_TQQQ, HEDGE, NO_CHANGE
    qlib_score: float
    confidence: float

class VixSqueezeQlibEngine:
    def __init__(self, qlib_model_path: str = None):
        self.forecaster = QlibForecaster(qlib_model_path)
        self.high_vix = 25.0
        self.low_vix = 15.0

    def analyze(self, vix_history_list: List[float]) -> QlibVixSignal:
        # Convert list to DF for the neural network
        history_df = pd.DataFrame(vix_history_list, columns=['close'])
        curr_vix = vix_history_list[-1]
        
        # 1. Get Qlib Prediction Score
        score = self.forecaster.get_vix_score(history_df)
        
        # 2. Calculate Momentum (Same as before)
        slope_3d = (vix_history_list[-1] - vix_history_list[-3]) / vix_history_list[-3] * 100 if len(vix_history_list) > 3 else 0
        
        action = "NO_CHANGE"
        confidence = 0.5

        # --- QLIB-CONFIRMED VIX SQUEEZE STRATEGY ---
        # 1. Spike Protection: if VIX slope suddenly jumps, exit regardless of level.
        if slope_3d > 1.5:
            action = "HEDGE"
            confidence = 0.95
            
        # 2. Squeeze Buy: extreme fear, but model sees VIX topping and rolling over.
        elif curr_vix > self.high_vix:
            if score < 0.3: # Qlib predicts VIX will drop
                action = "LONG_TQQQ"
                confidence = 0.9
            else:
                action = "NO_CHANGE"
        
        # 3. Complacency Exit: low VIX, but model sees VIX bottoming and rising.
        elif curr_vix < self.low_vix:
            if score > 0.7: # Qlib predicts VIX will spike
                action = "HEDGE"
                confidence = 0.85
            else:
                action = "NO_CHANGE"
                
        return QlibVixSignal(
            vix_current=curr_vix,
            action=action,
            qlib_score=score,
            confidence=confidence
        )
