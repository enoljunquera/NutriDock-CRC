import argparse
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Optional: torch (solo si usas --model mlp)
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
        y3 = df["y"].to_numpy(dtype=int)
    else:
        if "Diagnostico" in df.columns:
            y3 = df["Diagnostico"].map(LABEL_MAP).to_numpy(dtype=int)
        elif "Diagnóstico" in df.columns:
            y3 = df["Diagnóstico"].map(LABEL_MAP).to_numpy(dtype=int)
        else:
            raise ValueError("No encuentro columna 'y' ni 'Diagnostico/Diagnóstico'.")

    feat_cols = [c for c in df.columns if c.startswith(("DietW__", "Diet__total_grams", "MolAgg__", "ProtAgg__"))]
    if not feat_cols:
        raise ValueError("No encuentro columnas de features esperadas.")

    X = df[feat_cols].to_numpy(dtype=np.float32)

    # binario: 0 normal, 1 alterado (polipo+neoplasia)
    y_bin = (y3 != 0).astype(int)
    return df, X, y3, y_bin, feat_cols


def pick_threshold(y_true, p1, mode="youden", target_spec=0.90):
    """
    y_true: {0,1}
    p1: probabilidad de clase 1 (alterado)
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    best = None

    for t in thresholds:
        y_pred = (p1 >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        bal = 0.5 * (spec + rec)

        if mode == "fixed_spec":
            if spec < target_spec:
                continue
            # maximiza recall y luego bal
            key = (rec, bal)
        else:
            # youden/balanced
            key = (bal, rec)

        if best is None or key > best["key"]:
            best = {"t": float(t), "spec": float(spec), "rec": float(rec), "bal": float(bal), "key": key}

    # si fixed_spec no encuentra nada, cae a youden
    if best is None:
        return pick_threshold(y_true, p1, mode="youden", target_spec=target_spec)

    return best


# ------------------ Models ------------------

def make_hgb(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_depth=6,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=30,
        random_state=seed
    )


def make_logreg(seed: int):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        random_state=seed
    )


class BinMLP(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def class_weights_bin(y_train):
    # balanceado
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def fit_mlp_inner_best_epoch(Xtr, ytr, inner_val_frac, max_epochs, patience,
                            batch_size, lr, weight_decay, dropout, seed, device):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=inner_val_frac, random_state=seed)
    in_tr, in_va = next(sss.split(Xtr, ytr))
    X_in, y_in = Xtr[in_tr], ytr[in_tr]
    X_iv, y_iv = Xtr[in_va], ytr[in_va]

    X_in_t = torch.tensor(X_in, dtype=torch.float32, device=device)
    y_in_t = torch.tensor(y_in, dtype=torch.long, device=device)
    X_iv_t = torch.tensor(X_iv, dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(X_in_t, y_in_t), batch_size=batch_size, shuffle=True)

    model = BinMLP(in_dim=Xtr.shape[1], dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = class_weights_bin(y_in).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    best_bal = -1.0
    best_epoch = 1
    bad = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X_iv_t), dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)
            bal = balanced_accuracy_score(y_iv, pred)

        if bal > best_bal + 1e-4:
            best_bal = bal
            best_epoch = ep
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_epoch


def fit_mlp_fixed_epochs(Xtr, ytr, epochs, batch_size, lr, weight_decay, dropout, seed, device):
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)

    model = BinMLP(in_dim=Xtr.shape[1], dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = class_weights_bin(ytr).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    return model


def mlp_predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=next(model.parameters()).device)
        p = torch.softmax(model(X_t), dim=1).cpu().numpy()
    return p


# ------------------ Experiment ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="combined_features.csv")
    ap.add_argument("--model", type=str, default="hgb", choices=["hgb", "logreg", "mlp"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--thr_mode", type=str, default="youden", choices=["youden", "fixed_spec"])
    ap.add_argument("--target_spec", type=float, default=0.90)

    # MLP params (si usas --model mlp)
    ap.add_argument("--inner_val_frac", type=float, default=0.2)
    ap.add_argument("--max_epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.30)

    ap.add_argument("--out_prefix", type=str, default="exp_bin")
    args = ap.parse_args()

    df, X, y3, y, feat_cols = load_combined(args.csv)
    print("Loaded:", args.csv, "X:", X.shape, "y_bin:", y.shape)
    u, c = np.unique(y, return_counts=True)
    print("Binary class counts:", dict(zip(u.tolist(), c.tolist())), "(0=Normal, 1=Alterado)")

    rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)

    split_rows = []
    oof_rows = []
    split_id = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for tr_idx, te_idx in rskf.split(X, y):
        split_id += 1
        split_seed = args.seed + split_id
        set_seed(split_seed)

        Xtr_raw, Xte_raw = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        # Scaling para logreg/mlp. HGB no lo necesita, pero no le suele molestar.
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_raw).astype(np.float32)
        Xte = scaler.transform(Xte_raw).astype(np.float32)

        # Inner split para umbral (y para best_epoch en MLP)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.inner_val_frac, random_state=split_seed)
        in_tr, in_va = next(sss.split(Xtr, ytr))
        X_in, y_in = Xtr[in_tr], ytr[in_tr]
        X_iv, y_iv = Xtr[in_va], ytr[in_va]

        if args.model == "hgb":
            model = make_hgb(split_seed)
            model.fit(X_in, y_in)
            p_iv = model.predict_proba(X_iv)[:, 1]
            thr = pick_threshold(y_iv, p_iv, mode=args.thr_mode, target_spec=args.target_spec)

            # retrain en todo train del fold
            model2 = make_hgb(split_seed)
            model2.fit(Xtr, ytr)
            p_te = model2.predict_proba(Xte)[:, 1]

        elif args.model == "logreg":
            model = make_logreg(split_seed)
            model.fit(X_in, y_in)
            p_iv = model.predict_proba(X_iv)[:, 1]
            thr = pick_threshold(y_iv, p_iv, mode=args.thr_mode, target_spec=args.target_spec)

            model2 = make_logreg(split_seed)
            model2.fit(Xtr, ytr)
            p_te = model2.predict_proba(Xte)[:, 1]

        else:  # mlp
            best_epoch = fit_mlp_inner_best_epoch(
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

            # umbral usando inner-val con modelo entrenado SOLO en inner-tr (rápido)
            # entrenamos un modelo pequeño con best_epoch sobre X_in, y elegimos thr en X_iv
            m_tmp = fit_mlp_fixed_epochs(
                Xtr=X_in, ytr=y_in,
                epochs=best_epoch,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                seed=split_seed,
                device=device
            )
            p_iv = mlp_predict_proba(m_tmp, X_iv)[:, 1]
            thr = pick_threshold(y_iv, p_iv, mode=args.thr_mode, target_spec=args.target_spec)

            # retrain en todo train del fold con best_epoch y eval en test
            model2 = fit_mlp_fixed_epochs(
                Xtr=Xtr, ytr=ytr,
                epochs=best_epoch,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                seed=split_seed,
                device=device
            )
            p_te = mlp_predict_proba(model2, Xte)[:, 1]

        y_pred = (p_te >= thr["t"]).astype(int)

        # métricas
        tn, fp, fn, tp = confusion_matrix(yte, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0

        bal = balanced_accuracy_score(yte, y_pred)
        acc = accuracy_score(yte, y_pred)
        f1 = f1_score(yte, y_pred, zero_division=0)

        # métricas de ranking (no dependen del umbral)
        try:
            auroc = roc_auc_score(yte, p_te)
        except Exception:
            auroc = float("nan")
        try:
            auprc = average_precision_score(yte, p_te)
        except Exception:
            auprc = float("nan")

        split_rows.append({
            "split_id": split_id,
            "model": args.model,
            "thr_mode": args.thr_mode,
            "thr": thr["t"],
            "inner_spec": thr["spec"],
            "inner_recall": thr["rec"],
            "bal_acc": bal,
            "acc": acc,
            "f1": f1,
            "recall_abnormal": rec,
            "specificity_normal": spec,
            "auroc": auroc,
            "auprc": auprc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        })

        for local_i, global_i in enumerate(te_idx):
            oof_rows.append({
                "split_id": split_id,
                "idx": int(global_i),
                "y_true_bin": int(y[global_i]),
                "p_abnormal": float(p_te[local_i]),
                "y_pred_bin": int(y_pred[local_i]),
                "thr": float(thr["t"])
            })

        print(f"[split {split_id:03d}] thr={thr['t']:.2f} bal={bal:.3f} rec={rec:.3f} spec={spec:.3f} auroc={auroc:.3f} auprc={auprc:.3f}")

    split_df = pd.DataFrame(split_rows)
    oof_df = pd.DataFrame(oof_rows)

    split_csv = f"{args.out_prefix}_{args.model}_{args.thr_mode}_split_metrics.csv"
    oof_csv = f"{args.out_prefix}_{args.model}_{args.thr_mode}_oof.csv"
    split_df.to_csv(split_csv, index=False)
    oof_df.to_csv(oof_csv, index=False)

    # resumen
    def ms(col):
        return float(np.nanmean(split_df[col])), float(np.nanstd(split_df[col], ddof=1))

    bal_m, bal_s = ms("bal_acc")
    rec_m, rec_s = ms("recall_abnormal")
    spec_m, spec_s = ms("specificity_normal")
    auroc_m, auroc_s = ms("auroc")
    auprc_m, auprc_s = ms("auprc")

    print("\n=== SUMMARY (mean ± std across splits) ===")
    print(f"balanced acc: {bal_m:.4f} ± {bal_s:.4f}")
    print(f"recall(abnormal): {rec_m:.4f} ± {rec_s:.4f}")
    print(f"specificity(normal): {spec_m:.4f} ± {spec_s:.4f}")
    print(f"AUROC: {auroc_m:.4f} ± {auroc_s:.4f}")
    print(f"AUPRC: {auprc_m:.4f} ± {auprc_s:.4f}")
    print(f"\nSaved:\n  {split_csv}\n  {oof_csv}")


if __name__ == "__main__":
    main()
