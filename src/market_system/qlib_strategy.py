import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle

class VixGRU(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, num_layers=2):
        super(VixGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class QlibForecaster:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = "vix_gru_model.pth"
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Resolve absolute path for the model
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), model_path)
        model_dir = os.path.dirname(model_path)
        self.model_path = model_path
        self.is_available = False
        try:
            with open(os.path.join(model_dir, "vix_scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(model_dir, "vix_features.pkl"), "rb") as f:
                self.features = pickle.load(f)
                
            self.model = VixGRU(input_dim=len(self.features))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.is_available = True
            print(f"🔥 Real Qlib GRU Model Loaded Successfully.")
        except Exception as e:
            print(f"⚠️ Error loading real model: {e}. Falling back to baseline.")
            self.model = None

    def get_vix_score(self, history_df):
        """
        Input: DataFrame with 'close' column
        """
        if self.model is None or len(history_df) < 65:
            return 0.5
            
        # 1. Feature Engineering (Must match training)
        df = history_df.copy()
        for n in [5, 10, 20, 60]:
            df[f'roc_{n}'] = df['close'].pct_change(n)
            df[f'ma_{n}'] = df['close'] / df['close'].rolling(n).mean()
            df[f'std_{n}'] = df['close'].rolling(n).std() / df['close']
            
        df = df.dropna()
        if len(df) < 30: return 0.5
        
        # 2. Scale and Sequence
        X = self.scaler.transform(df[self.features].values)
        X_seq = torch.FloatTensor(X[-30:]).unsqueeze(0).to(self.device)
        
        # 3. Predict
        with torch.no_grad():
            score = self.model(X_seq).item()
            
        # The model is trained with BCEWithLogitsLoss, so the raw output is a logit.
        prob_spike = 1 / (1 + np.exp(-score))
        return prob_spike
