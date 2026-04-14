import argparse
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix


LABEL_MAP = {"Normal": 0, "Pólipo": 1, "Neoplasia": 2}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_combined(csv_path: str):
    df = pd.read_csv(csv_path)

    if "y" in df.columns:
        y = df["y"].to_numpy(dtype=int)
    else:
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


def compute_sample_weights(y_train: np.ndarray, num_classes: int = 3):
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    return w[y_train].astype(np.float32)


def get_booster(kind_preference="auto"):
    """
    Devuelve (kind, factory) donde factory(train_seed) -> model
    kind en {"catboost", "xgboost", "sk_hgb"}
    """
    # 1) CatBoost
    try:
        from catboost import CatBoostClassifier

        def make_cat(seed):
            return CatBoostClassifier(
                loss_function="MultiClass",
                iterations=5000,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=seed,
                verbose=False,
                auto_class_weights="Balanced",
            )

        return "catboost", make_cat
    except Exception:
        pass

    # 2) XGBoost
    try:
        from xgboost import XGBClassifier

        def make_xgb(seed):
            return XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=5000,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_weight=1.0,
                tree_method="hist",
                random_state=seed,
                eval_metric="mlogloss",
            )

        return "xgboost", make_xgb
    except Exception:
        pass

    # 3) Fallback sklearn HGB
    from sklearn.ensemble import HistGradientBoostingClassifier

    def make_hgb(seed):
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=6,
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=30,
            random_state=seed
        )

    return "sk_hgb", make_hgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="combined_features.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--inner_val_frac", type=float, default=0.2)
    ap.add_argument("--early_stop_rounds", type=int, default=50)  # para cat/xgb
    ap.add_argument("--out_prefix", type=str, default="exp3_boosting_repeated")
    args = ap.parse_args()

    df, X, y, feat_cols = load_combined(args.csv)
    print("Loaded:", args.csv, "X:", X.shape, "y:", y.shape)
    u, c = np.unique(y, return_counts=True)
    print("Class counts:", dict(zip(u.tolist(), c.tolist())))

    kind, make_model = get_booster()
    print("Using booster:", kind)

    rskf = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.seed
    )

    oof_rows = []
    split_rows = []

    split_id = 0
    for tr_idx, te_idx in rskf.split(X, y):
        split_id += 1
        split_seed = args.seed + split_id
        set_seed(split_seed)

        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        # inner split SOLO para early stopping en boosters que lo soportan
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.inner_val_frac, random_state=split_seed)
        in_tr, in_va = next(sss.split(Xtr, ytr))
        X_in, y_in = Xtr[in_tr], ytr[in_tr]
        X_iv, y_iv = Xtr[in_va], ytr[in_va]

        sw_in = compute_sample_weights(y_in, 3)

        model = make_model(split_seed)

        if kind == "catboost":
            model.fit(X_in, y_in, eval_set=(X_iv, y_iv), use_best_model=True, early_stopping_rounds=args.early_stop_rounds)
            best_it = int(getattr(model, "best_iteration_", model.get_best_iteration()))
            # retrain en todo el train del fold con best_it
            model2 = make_model(split_seed)
            model2.set_params(iterations=max(best_it, 1))
            sw_tr = compute_sample_weights(ytr, 3)
            model2.fit(Xtr, ytr, sample_weight=sw_tr, verbose=False)
            model = model2

        elif kind == "xgboost":
            model.fit(
                X_in, y_in,
                sample_weight=sw_in,
                eval_set=[(X_iv, y_iv)],
                verbose=False,
                early_stopping_rounds=args.early_stop_rounds
            )
            best_it = int(model.best_iteration + 1) if hasattr(model, "best_iteration") and model.best_iteration is not None else model.get_params().get("n_estimators", 1000)
            model2 = make_model(split_seed)
            model2.set_params(n_estimators=max(best_it, 1))
            sw_tr = compute_sample_weights(ytr, 3)
            model2.fit(Xtr, ytr, sample_weight=sw_tr, verbose=False)
            model = model2

        else:
            # sklearn HGB: early stopping interno, podemos pasar sample_weight en fit
            sw_tr = compute_sample_weights(ytr, 3)
            model.fit(Xtr, ytr, sample_weight=sw_tr)

        probs = model.predict_proba(Xte)
        pred = probs.argmax(axis=1)

        f1m = f1_score(yte, pred, average="macro", zero_division=0)
        bacc = balanced_accuracy_score(yte, pred)
        acc = accuracy_score(yte, pred)
        cm = confusion_matrix(yte, pred, labels=[0, 1, 2])

        print(f"[split {split_id:03d}] macroF1={f1m:.3f}  balAcc={bacc:.3f}  acc={acc:.3f}")

        split_rows.append({
            "split_id": split_id,
            "macro_f1": f1m,
            "bal_acc": bacc,
            "acc": acc,
            "cm_00": cm[0,0], "cm_01": cm[0,1], "cm_02": cm[0,2],
            "cm_10": cm[1,0], "cm_11": cm[1,1], "cm_12": cm[1,2],
            "cm_20": cm[2,0], "cm_21": cm[2,1], "cm_22": cm[2,2],
        })

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

    split_csv = f"{args.out_prefix}_{kind}_split_metrics.csv"
    oof_csv = f"{args.out_prefix}_{kind}_oof.csv"
    split_df.to_csv(split_csv, index=False)
    oof_df.to_csv(oof_csv, index=False)

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
