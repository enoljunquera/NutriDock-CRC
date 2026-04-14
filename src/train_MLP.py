# train_MLP.py
import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y_train: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------
# Tabular model
# ---------------------------
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


# ---------------------------
# Optional: Graph encoder (torch_geometric)
# ---------------------------
def try_import_pyg():
    try:
        from torch_geometric.loader import DataLoader as GeoDataLoader
        from torch_geometric.nn import GINConv, global_mean_pool
        return GeoDataLoader, GINConv, global_mean_pool
    except Exception:
        return None, None, None


class SmallGIN(nn.Module):
    """
    Encoder pequeño para Data/Batch(x, edge_index, batch).
    Devuelve embedding (num_graphs, emb_dim).
    """
    def __init__(self, in_dim: int, emb_dim: int = 64, dropout: float = 0.10,
                 GINConv=None, global_mean_pool=None):
        super().__init__()
        assert GINConv is not None and global_mean_pool is not None
        self.global_mean_pool = global_mean_pool
        self.dropout = dropout

        def mlp(inf, outf):
            return nn.Sequential(
                nn.Linear(inf, outf),
                nn.ReLU(),
                nn.Linear(outf, outf),
            )

        self.conv1 = GINConv(mlp(in_dim, emb_dim))
        self.conv2 = GINConv(mlp(emb_dim, emb_dim))
        self.lin = nn.Linear(emb_dim, emb_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.global_mean_pool(x, batch)
        x = self.lin(x)
        return x


class TabularPlusGraphNet(nn.Module):
    """
    - GNN produce embeddings por CID: E (37, emb_dim)
    - Para cada sujeto: GraphAgg = grams(37) @ E
    - Concatena [tab_features, GraphAgg] -> MLP final
    """
    def __init__(self, tab_in_dim: int, graph_in_dim: int, emb_dim: int = 64,
                 dropout_tab=0.30, dropout_gnn=0.10, GINConv=None, global_mean_pool=None):
        super().__init__()
        self.gnn = SmallGIN(graph_in_dim, emb_dim=emb_dim, dropout=dropout_gnn,
                            GINConv=GINConv, global_mean_pool=global_mean_pool)

        self.mlp = nn.Sequential(
            nn.Linear(tab_in_dim + emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_tab),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_tab),
            nn.Linear(32, 3),
        )

    def forward(self, tab_x, grams_x, cid_embs):
        # cid_embs: (37, emb_dim)  grams_x: (B, 37)
        graph_agg = grams_x @ cid_embs  # (B, emb_dim)
        x = torch.cat([tab_x, graph_agg], dim=1)
        return self.mlp(x)


# ---------------------------
# Data loading (combined CSV)
# ---------------------------
def load_combined(csv_path: str):
    df = pd.read_csv(csv_path)

    # y
    if "y" in df.columns:
        y = df["y"].to_numpy(dtype=int)
    else:
        label_map = {"Normal": 0, "Pólipo": 1, "Neoplasia": 2}
        if "Diagnostico" in df.columns:
            y = df["Diagnostico"].map(label_map).to_numpy(dtype=int)
        elif "Diagnóstico" in df.columns:
            y = df["Diagnóstico"].map(label_map).to_numpy(dtype=int)
        else:
            raise ValueError("No encuentro columna 'y' ni 'Diagnostico/Diagnóstico' en el CSV.")

    # features tabulares
    feat_cols = [c for c in df.columns if c.startswith(("DietW__", "Diet__total_grams", "MolAgg__", "ProtAgg__"))]
    if len(feat_cols) == 0:
        raise ValueError("No encuentro columnas de features esperadas (DietW__, Diet__total_grams, MolAgg__, ProtAgg__).")
    X = df[feat_cols].to_numpy(dtype=np.float32)

    # reconstruir grams por CID desde DietW__* y total grams:
    dietw_cols = [c for c in df.columns if c.startswith("DietW__")]
    if "Diet__total_grams" not in df.columns:
        raise ValueError("Falta Diet__total_grams en el CSV (necesaria para grams).")
    W = df[dietw_cols].to_numpy(dtype=np.float32)  # (n,37)
    G = df["Diet__total_grams"].to_numpy(dtype=np.float32).reshape(-1, 1)  # (n,1)
    grams = (W * G).astype(np.float32)  # (n,37)

    # extraer CIDs del nombre DietW__CID
    cids = [int(c.split("__")[1]) for c in dietw_cols]

    return df, X, y, feat_cols, grams, cids


# ---------------------------
# Graph bank loading (3 contenedores .pt)
# ---------------------------
def _to_int_cid(k):
    try:
        return int(k)
    except Exception:
        return None


def load_graph_bank_from_dir(graphs_dir: str):
    """
    Carga todos los .pt dentro de 'graphs_dir' y construye:
      bank[cid] = Data
    Soporta:
      - dict: {cid: Data}
      - list/tuple: [Data,...] con:
          A) txt asociado en graphs_txt/<stem>.txt con CIDs
          B) atributo g.cid / g.CID / g.pubchem_cid / g.PubChemCID
    """
    gdir = Path(graphs_dir)
    if not gdir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {graphs_dir}")

    pt_files = sorted(gdir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No encuentro .pt en {graphs_dir}")

    bank = {}
    txt_dir = gdir / "graphs_txt"

    for fp in pt_files:
        obj = torch.load(fp, map_location="cpu")

        # Caso 1: dict {CID: Data}
        if isinstance(obj, dict):
            for k, v in obj.items():
                cid = _to_int_cid(k)
                if cid is not None:
                    bank[cid] = v
            continue

        # Caso 2: lista/tupla [Data,...]
        if isinstance(obj, (list, tuple)):
            # Intento A: txt con CIDs
            cand = txt_dir / (fp.stem + ".txt")
            if cand.exists():
                cids = []
                with open(cand, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        m = re.findall(r"\d+", line)
                        if m:
                            cids.append(int(m[0]))
                if len(cids) != len(obj):
                    raise ValueError(
                        f"El txt {cand} tiene {len(cids)} CIDs pero {fp.name} trae {len(obj)} grafos."
                    )
                for cid, g in zip(cids, obj):
                    bank[cid] = g
                continue

            # Intento B: atributo dentro de cada Data
            ok = 0
            for g in obj:
                cid = None
                for attr in ("cid", "CID", "pubchem_cid", "PubChemCID"):
                    if hasattr(g, attr):
                        cid = _to_int_cid(getattr(g, attr))
                        break
                if cid is not None:
                    bank[cid] = g
                    ok += 1

            if ok == 0:
                raise TypeError(
                    f"No pude mapear los grafos de {fp.name} a CIDs.\n"
                    f"No hay dict, no hay txt asociado en {txt_dir}, y los Data no tienen atributo cid/CID."
                )
            continue

        raise TypeError(f"Tipo no soportado en {fp.name}: {type(obj)}")

    return bank


def load_graphs_for_cids(cids, graphs_dir: str):
    bank = load_graph_bank_from_dir(graphs_dir)
    missing = [cid for cid in cids if cid not in bank]
    if missing:
        some = sorted(list(bank.keys()))[:20]
        raise FileNotFoundError(
            f"Faltan grafos para estos CIDs: {missing}\n"
            f"Ejemplo de CIDs disponibles en bank: {some}"
        )
    return [bank[cid] for cid in cids]


# ---------------------------
# CV trainer
# ---------------------------
def run_5fold_cv(
    X: np.ndarray,
    y: np.ndarray,
    grams: np.ndarray,
    use_graphs: bool,
    graphs,
    graph_in_dim,
    seed: int = 42,
    epochs: int = 300,
    patience: int = 30,
    batch_size: int = 32,
    lr: float = 2e-3,
    weight_decay: float = 1e-2,
    dropout: float = 0.30,
    out_prefix: str = "day2"
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    oof_probs = np.zeros((len(y), 3), dtype=np.float32)

    GeoDataLoader, GINConv, global_mean_pool = try_import_pyg()
    if use_graphs and GeoDataLoader is None:
        raise RuntimeError("use_graphs=True pero no puedo importar torch_geometric.")

    # Prepara un único batch con los 37 grafos (orden fijo)
    if use_graphs:
        geo_loader = GeoDataLoader(graphs, batch_size=len(graphs), shuffle=False)
        graph_batch_cpu = next(iter(geo_loader))  # Batch con todos los grafos
    else:
        graph_batch_cpu = None

    fold_metrics = []

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        set_seed(seed + fold)

        # --- scaler por fold (sin leakage) ---
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr]).astype(np.float32)
        Xva = scaler.transform(X[va]).astype(np.float32)

        # tensores
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(device)
        ytr_t = torch.tensor(y[tr], dtype=torch.long).to(device)
        Xva_t = torch.tensor(Xva, dtype=torch.float32).to(device)

        grams_tr_t = torch.tensor(grams[tr], dtype=torch.float32).to(device)
        grams_va_t = torch.tensor(grams[va], dtype=torch.float32).to(device)

        loader = DataLoader(
            TensorDataset(Xtr_t, grams_tr_t, ytr_t),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        # modelo
        if not use_graphs:
            model = SmallMLP(in_dim=X.shape[1], num_classes=3, dropout=dropout).to(device)
        else:
            model = TabularPlusGraphNet(
                tab_in_dim=X.shape[1],
                graph_in_dim=graph_in_dim,
                emb_dim=64,
                dropout_tab=dropout,
                dropout_gnn=0.10,
                GINConv=GINConv,
                global_mean_pool=global_mean_pool,
            ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        w = class_weights(y[tr], num_classes=3).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=w)

        best_f1 = -1.0
        best_state = None
        bad = 0

        # mueve el batch de grafos al device una vez por fold
        if use_graphs:
            graph_batch = graph_batch_cpu.to(device)

        for ep in range(epochs):
            model.train()

            for xb, gramsb, yb in loader:
                opt.zero_grad()

                if not use_graphs:
                    logits = model(xb)
                else:
                    # ✅ FIX: recalcular embeddings en CADA mini-batch (evita backward 2 veces)
                    cid_embs = model.gnn(graph_batch)      # (37, emb_dim) con gradientes
                    logits = model(xb, gramsb, cid_embs)

                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

            # --- validación fold ---
            model.eval()
            with torch.no_grad():
                if not use_graphs:
                    probs = torch.softmax(model(Xva_t), dim=1).cpu().numpy()
                else:
                    cid_embs_eval = model.gnn(graph_batch)  # sin grad
                    probs = torch.softmax(model(Xva_t, grams_va_t, cid_embs_eval), dim=1).cpu().numpy()

                pred = probs.argmax(axis=1)
                f1 = f1_score(y[va], pred, average="macro", zero_division=0)

            if f1 > best_f1 + 1e-4:
                best_f1 = f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        # cargar best y guardar oof
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            if not use_graphs:
                probs = torch.softmax(model(Xva_t), dim=1).cpu().numpy()
            else:
                cid_embs_eval = model.gnn(graph_batch)
                probs = torch.softmax(model(Xva_t, grams_va_t, cid_embs_eval), dim=1).cpu().numpy()

        oof_probs[va] = probs
        oof_pred = probs.argmax(axis=1)

        fold_f1 = f1_score(y[va], oof_pred, average="macro", zero_division=0)
        fold_ba = balanced_accuracy_score(y[va], oof_pred)
        fold_acc = accuracy_score(y[va], oof_pred)

        fold_metrics.append((fold_f1, fold_ba, fold_acc))
        print(f"[fold {fold}] best macro-F1={best_f1:.3f} | fold macro-F1={fold_f1:.3f} bal_acc={fold_ba:.3f} acc={fold_acc:.3f} (stopped ep {ep+1})")

    # métricas OOF globales
    oof_pred_all = oof_probs.argmax(axis=1)
    macro_f1 = f1_score(y, oof_pred_all, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(y, oof_pred_all)
    acc      = accuracy_score(y, oof_pred_all)
    cm       = confusion_matrix(y, oof_pred_all)

    print("\nOOF results:")
    print("macro-F1:", float(macro_f1))
    print("bal acc :", float(bal_acc))
    print("acc     :", float(acc))
    print("confusion matrix:\n", cm)

    # guardar
    np.save(f"{out_prefix}_oof_probs.npy", oof_probs)
    np.save(f"{out_prefix}_oof_pred.npy", oof_pred_all)

    with open(f"{out_prefix}_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"use_graphs={use_graphs}\n")
        f.write(f"macro_f1={macro_f1}\n")
        f.write(f"bal_acc={bal_acc}\n")
        f.write(f"acc={acc}\n")
        f.write("fold_metrics (macroF1, balAcc, acc):\n")
        for i, (ff1, fba, facc) in enumerate(fold_metrics, start=1):
            f.write(f"  fold{i}: {ff1:.4f}, {fba:.4f}, {facc:.4f}\n")
        f.write("confusion_matrix:\n")
        f.write(str(cm))

    print(f"\nSaved: {out_prefix}_oof_probs.npy, {out_prefix}_oof_pred.npy, {out_prefix}_metrics.txt")


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="combined_features.csv")
    ap.add_argument("--graphs_dir", type=str, default="Grafos")
    ap.add_argument("--use_graphs", action="store_true", help="Añade GraphAgg (requiere torch_geometric).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--out_prefix", type=str, default="day2")
    args = ap.parse_args()

    df, X, y, feat_cols, grams, cids = load_combined(args.csv)
    print("Loaded:", args.csv)
    print("X shape:", X.shape, "y shape:", y.shape)
    unique, counts = np.unique(y, return_counts=True)
    print("Class counts:", dict(zip(unique.tolist(), counts.tolist())))

    graphs = None
    graph_in_dim = None

    if args.use_graphs:
        GeoDataLoader, GINConv, global_mean_pool = try_import_pyg()
        if GeoDataLoader is None:
            raise RuntimeError("No se puede importar torch_geometric. Instálalo o ejecuta sin --use_graphs.")

        graphs = load_graphs_for_cids(cids, args.graphs_dir)

        x0 = graphs[0].x
        if x0 is None:
            raise ValueError("El grafo cargado no tiene atributo 'x' (node features).")
        graph_in_dim = int(x0.shape[1])

        print(f"Graphs loaded: {len(graphs)} | graph_in_dim={graph_in_dim}")

    run_5fold_cv(
        X=X,
        y=y,
        grams=grams,
        use_graphs=args.use_graphs,
        graphs=graphs,
        graph_in_dim=graph_in_dim,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        out_prefix=args.out_prefix
    )


if __name__ == "__main__":
    main()
