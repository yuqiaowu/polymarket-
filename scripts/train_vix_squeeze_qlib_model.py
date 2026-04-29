import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from market_system.qlib_strategy import VixGRU


FEATURES = [
    "roc_5",
    "ma_5",
    "std_5",
    "roc_10",
    "ma_10",
    "std_10",
    "roc_20",
    "ma_20",
    "std_20",
    "roc_60",
    "ma_60",
    "std_60",
]


def main():
    args = _parse_args()
    _set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"Fetching VIX daily data from {args.start} to {args.train_end}...")
    raw = yf.download("^VIX", start=args.start, end=args.train_end, auto_adjust=True, progress=False)
    close = _close_series(raw, "^VIX")
    dataset = _build_dataset(close, args.sequence_length, args.horizon, args.spike_threshold)
    if len(dataset["X"]) < 200:
        raise RuntimeError(f"Not enough training samples: {len(dataset['X'])}")

    train_loader, valid_loader, scaler, split_info = _prepare_loaders(dataset, args.sequence_length, args.batch_size)
    model = VixGRU(input_dim=len(FEATURES), hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    metrics = _train(model, train_loader, valid_loader, args.epochs, args.learning_rate)

    model_path = os.path.join(args.model_dir, "vix_gru_model.pth")
    scaler_path = os.path.join(args.model_dir, "vix_scaler.pkl")
    features_path = os.path.join(args.model_dir, "vix_features.pkl")
    metadata_path = os.path.join(args.model_dir, "vix_gru_metadata.pkl")

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)
    with open(features_path, "wb") as file:
        pickle.dump(FEATURES, file)
    with open(metadata_path, "wb") as file:
        pickle.dump(
            {
                "start": args.start,
                "train_end": args.train_end,
                "sequence_length": args.sequence_length,
                "horizon": args.horizon,
                "spike_threshold": args.spike_threshold,
                "features": FEATURES,
                "split": split_info,
                "metrics": metrics,
            },
            file,
        )

    print("\n=== VIX GRU TRAINING COMPLETE ===")
    print(f"Samples: {split_info['samples']} | Train: {split_info['train_samples']} | Valid: {split_info['valid_samples']}")
    print(f"Positive spike rate: {split_info['positive_rate']:.2%}")
    print(f"Validation loss: {metrics['valid_loss']:.4f}")
    print(f"Validation accuracy: {metrics['valid_accuracy']:.2%}")
    print(f"Validation AUC: {metrics['valid_auc']:.3f}")
    print(f"Saved model to: {os.path.abspath(args.model_dir)}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the GRU forecaster used by the VIX squeeze Qlib-style strategy.")
    parser.add_argument("--start", default="2005-01-01", help="Training history start date.")
    parser.add_argument("--train-end", default="2022-01-01", help="Exclusive training end date. Keep before backtest start to avoid leakage.")
    parser.add_argument("--model-dir", default="models/vix_gru", help="Output directory for model and scaler files.")
    parser.add_argument("--sequence-length", type=int, default=30, help="Daily feature rows per model sample.")
    parser.add_argument("--horizon", type=int, default=5, help="Forward VIX horizon in trading days.")
    parser.add_argument("--spike-threshold", type=float, default=0.05, help="Future VIX return threshold for spike label.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _close_series(frame, symbol):
    if frame.empty:
        raise RuntimeError(f"No data downloaded for {symbol}")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.astype(float)


def _build_dataset(close, sequence_length, horizon, spike_threshold):
    df = pd.DataFrame({"close": close})
    for n in [5, 10, 20, 60]:
        df[f"roc_{n}"] = df["close"].pct_change(n)
        df[f"ma_{n}"] = df["close"] / df["close"].rolling(n).mean()
        df[f"std_{n}"] = df["close"].rolling(n).std() / df["close"]
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["target"] = (df["future_return"] > spike_threshold).astype(float)
    df = df.dropna()

    samples = []
    targets = []
    dates = []
    values = df[FEATURES].values.astype(np.float32)
    labels = df["target"].values.astype(np.float32)
    for end_idx in range(sequence_length - 1, len(df)):
        samples.append(values[end_idx - sequence_length + 1: end_idx + 1])
        targets.append(labels[end_idx])
        dates.append(df.index[end_idx])
    return {"X": np.asarray(samples, dtype=np.float32), "y": np.asarray(targets, dtype=np.float32), "dates": dates}


def _prepare_loaders(dataset, sequence_length, batch_size):
    X = dataset["X"]
    y = dataset["y"]
    split_idx = int(len(X) * 0.8)
    X_train, X_valid = X[:split_idx], X[split_idx:]
    y_train, y_valid = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(-1, sequence_length, X_train.shape[-1])
    X_valid = scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(-1, sequence_length, X_valid.shape[-1])

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_valid), torch.FloatTensor(y_valid).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=False,
    )
    split_info = {
        "samples": len(X),
        "train_samples": len(X_train),
        "valid_samples": len(X_valid),
        "positive_rate": float(y.mean()),
        "valid_start": dataset["dates"][split_idx].strftime("%Y-%m-%d"),
        "valid_end": dataset["dates"][-1].strftime("%Y-%m-%d"),
    }
    return train_loader, valid_loader, scaler, split_info


def _train(model, train_loader, valid_loader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_valid_loss = float("inf")
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        metrics = _evaluate(model, valid_loader, criterion)
        if metrics["valid_loss"] < best_valid_loss:
            best_valid_loss = metrics["valid_loss"]
            best_metrics = metrics
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:03d} "
                f"valid_loss={metrics['valid_loss']:.4f} "
                f"valid_acc={metrics['valid_accuracy']:.2%} "
                f"valid_auc={metrics['valid_auc']:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics


def _evaluate(model, valid_loader, criterion):
    model.eval()
    losses = []
    probabilities = []
    labels = []
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            losses.append(float(loss.item()))
            probabilities.extend(torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist())
            labels.extend(y_batch.squeeze(1).cpu().numpy().tolist())
    preds = [1.0 if prob >= 0.5 else 0.0 for prob in probabilities]
    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = 0.5
    return {
        "valid_loss": float(np.mean(losses)),
        "valid_accuracy": float(accuracy_score(labels, preds)),
        "valid_auc": float(auc),
    }


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
