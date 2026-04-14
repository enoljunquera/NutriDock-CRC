import argparse
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


LABEL_MAP = {"Normal": 0, "Pólipo": 1, "Neoplasia": 2}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_combined(csv_path: str):
    df = pd.read_csv(csv_path)

    if "y" in df.columns:
        y = df["y"].to_numpy(dtype=int)
    else:
        # fallback si solo tienes Diagnostico/Diagnóstico
        if "Diagnostico" in df.columns:
            y = df["Diagnostico"].map(LABEL_MAP).to_numpy(dtype=int)
        elif "Diagnóstico" in df.columns:
            y = df["Diagnóstico"].map(LABEL_MAP).to_numpy(dtype=int)
        else:
            raise ValueError("No encuentro columna 'y' ni 'Diagnostico/Diagnóstico'.")

    feat_cols = [c for c in df.columns if c.startswith(("DietW__", "Diet__total_grams", "MolAgg__", "ProtAgg__"))]
    if not feat_cols:
        raise ValueError("No encuentro columnas de features (DietW__, Diet__total_grams, MolAgg__, ProtAgg__).")

    X = df[feat_cols].to_numpy(dtype=np.float32)
    return df, X, y, feat_cols


def class_weights(y_train: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict_proba(model, X_t):
    model.eval()
    logits = model(X_t)
    return torch.softmax(logits, dim=1).cpu().numpy()


def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()


def fit_mlp_inner_earlystop(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    inner_val_frac: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    device
):
    """
    Devuelve best_epoch (>=1) usando inner split del TRAIN del fold.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=inner_val_frac, random_state=seed)
    inner_train_idx, inner_val_idx = next(sss.split(Xtr, ytr))

    X_in = Xtr[inner_train_idx]
    y_in = ytr[inner_train_idx]
    X_iv = Xtr[inner_val_idx]
    y_iv = ytr[inner_val_idx]

    X_in_t = torch.tensor(X_in, dtype=torch.float32, device=device)
    y_in_t = torch.tensor(y_in, dtype=torch.long, device=device)
    X_iv_t = torch.tensor(X_iv, dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(X_in_t, y_in_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    model = SmallMLP(in_dim=Xtr.shape[1], num_classes=3, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = class_weights(y_in, 3).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    best_f1 = -1.0
    best_epoch = 1
    bad = 0

    for ep in range(1, max_epochs + 1):
        train_one_epoch(model, loader, opt, loss_fn)

        probs = predict_proba(model, X_iv_t)
        pred = probs.argmax(axis=1)
        f1 = f1_score(y_iv, pred, average="macro", zero_division=0)

        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best_epoch = ep
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_epoch


def fit_mlp_fixed_epochs(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    device
):
    """
    Entrena en todo el TRAIN del fold durante epochs (sin usar el outer val).
    Devuelve el modelo entrenado.
    """
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)

    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    model = SmallMLP(in_dim=Xtr.shape[1], num_classes=3, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = class_weights(ytr, 3).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    for _ in range(epochs):
        train_one_epoch(model, loader, opt, loss_fn)

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="combined_features.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)  # caña real = 20
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--inner_val_frac", type=float, default=0.2)
    ap.add_argument("--max_epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=30)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.30)

    ap.add_argument("--out_prefix", type=str, default="exp1_mlp_repeated")
    args = ap.parse_args()

    df, X, y, feat_cols = load_combined(args.csv)
    print("Loaded:", args.csv, "X:", X.shape, "y:", y.shape)
    u, c = np.unique(y, return_counts=True)
    print("Class counts:", dict(zip(u.tolist(), c.tolist())))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rskf = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.seed
    )

    # OOF por repetición+fold
    oof_rows = []
    split_rows = []

    split_id = 0
    for tr_idx, te_idx in rskf.split(X, y):
        split_id += 1
        # Seed por split para reproducibilidad
        split_seed = args.seed + split_id
        set_seed(split_seed)

        Xtr_raw, Xte_raw = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        # Escalado SOLO con train del fold
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_raw).astype(np.float32)
        Xte = scaler.transform(Xte_raw).astype(np.float32)

        # 1) elegir best_epoch con inner early stopping (sin tocar outer test)
        best_epoch = fit_mlp_inner_earlystop(
            Xtr=Xtr, ytr=ytr,
            inner_val_frac=args.inner_val_frac,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            seed=split_seed,
            device=device
        )

        # 2) re-entrenar en todo el train del fold con best_epoch
        model = fit_mlp_fixed_epochs(
            Xtr=Xtr, ytr=ytr,
            epochs=best_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            seed=split_seed,
            device=device
        )

        # 3) evaluar en outer test fold
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
        probs = predict_proba(model, Xte_t)
        pred = probs.argmax(axis=1)

        f1m = f1_score(yte, pred, average="macro", zero_division=0)
        bacc = balanced_accuracy_score(yte, pred)
        acc = accuracy_score(yte, pred)
        cm = confusion_matrix(yte, pred, labels=[0, 1, 2])

        print(f"[split {split_id:03d}] best_epoch={best_epoch:03d}  macroF1={f1m:.3f}  balAcc={bacc:.3f}  acc={acc:.3f}")

        split_rows.append({
            "split_id": split_id,
            "best_epoch": best_epoch,
            "macro_f1": f1m,
            "bal_acc": bacc,
            "acc": acc,
            "cm_00": cm[0,0], "cm_01": cm[0,1], "cm_02": cm[0,2],
            "cm_10": cm[1,0], "cm_11": cm[1,1], "cm_12": cm[1,2],
            "cm_20": cm[2,0], "cm_21": cm[2,1], "cm_22": cm[2,2],
        })

        # OOF rows
        for local_i, global_i in enumerate(te_idx):
            oof_rows.append({
                "split_id": split_id,
                "idx": int(global_i),
                "y_true": int(y[global_i]),
                "p0": float(probs[local_i, 0]),
                "p1": float(probs[local_i, 1]),
                "p2": float(probs[local_i, 2]),
                "y_pred": int(pred[local_i])
            })

    split_df = pd.DataFrame(split_rows)
    oof_df = pd.DataFrame(oof_rows)

    split_csv = f"{args.out_prefix}_split_metrics.csv"
    oof_csv = f"{args.out_prefix}_oof.csv"
    split_df.to_csv(split_csv, index=False)
    oof_df.to_csv(oof_csv, index=False)

    # Resumen
    macro_mean = split_df["macro_f1"].mean()
    macro_std = split_df["macro_f1"].std(ddof=1)
    bal_mean = split_df["bal_acc"].mean()
    bal_std = split_df["bal_acc"].std(ddof=1)

    print("\n=== SUMMARY (across all splits) ===")
    print(f"macro-F1: {macro_mean:.4f} ± {macro_std:.4f}")
    print(f"bal acc : {bal_mean:.4f} ± {bal_std:.4f}")
    print(f"Saved:\n  {split_csv}\n  {oof_csv}")


if __name__ == "__main__":
    main()
