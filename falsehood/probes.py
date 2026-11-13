"""Step 3-4: train head probes three ways and measure what they detect.

Probe sets (one linear probe per head, the authors' recipe: Adam, lr 1e-3, batch 25, 25 epochs, BCE):
  A_original  authors' 2,612 dialogues, labels and per-dialogue split (sanity check vs their accuracies)
  A_grouped   authors' dialogues and labels, split by question
  B           all 4 cases x 2 templates, label 1 = sycophantic, split by question, class-balanced loss
Plus the faithful probes you already trained (--faithful_dir), evaluated as they are.

All probe sets are evaluated on the held-out questions' dialogues of every case. The decisive number is
the "falsehood-control" rate: how often a probe calls a non-sycophantic dialogue that ends on a false
answer sycophantic (challenged I->I, hypothesis C->I). A falsehood detector scores ~100% there.

Outputs in --out_dir: per_head.csv (every head x probe set), summary.csv, probes_*.pt, heatmaps.
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from extract import load_acts

SEED = 3407


def train_heads(X, y, Xval, yval, epochs=25, batch=25, lr=1e-3, pos_weight=None, seed=0, device="cpu"):
    """Train one independent logistic probe per (layer, head) at once.

    X: [n, L, H, D] (float16 ok), y: [n] in {0,1}. Equivalent to the authors' per-head nn.Linear(D, 1)
    trained with Adam/BCE, except that all heads see the same minibatch order.
    Returns W [L,H,D], b [L,H] (last epoch) and per-epoch validation metrics.
    """
    g = torch.Generator().manual_seed(seed)
    n, L, H, D = X.shape
    bound = 1 / D ** 0.5  # nn.Linear default init
    W = ((torch.rand(L, H, D, generator=g) * 2 - 1) * bound).to(device).requires_grad_()
    b = ((torch.rand(L, H, generator=g) * 2 - 1) * bound).to(device).requires_grad_()
    opt = torch.optim.Adam([W, b], lr=lr)
    X, y = X.to(device), y.float().to(device)
    Xval, yval = Xval.to(device), yval.float().to(device)
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=device)
    hist = {"acc": [], "bal_acc": []}
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            logits = torch.einsum("nlhd,lhd->nlh", X[idx].float(), W) + b
            target = y[idx][:, None, None].expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none").mean(0).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = predict(Xval, W, b)
            hist["acc"].append(((pred == yval[:, None, None]).float().mean(0)).cpu())
            hist["bal_acc"].append(balanced_acc(pred, yval).cpu())
    return W.detach().cpu(), b.detach().cpu(), {k: torch.stack(v) for k, v in hist.items()}


@torch.no_grad()
def predict(X, W, b, chunk=2048):
    W, b = W.to(X.device), b.to(X.device)
    out = [((torch.einsum("nlhd,lhd->nlh", X[s:s + chunk].float(), W) + b) > 0).float() for s in range(0, len(X), chunk)]
    return torch.cat(out)


def balanced_acc(pred, y):
    pos, neg = y == 1, y == 0
    return 0.5 * (pred[pos].mean(0) + (1 - pred[neg]).mean(0))


def case_rates(pred, meta_eval):
    """Fraction of dialogues called sycophantic, per case: {case: [L,H]}."""
    return {c: pred[torch.tensor((meta_eval["case"] == c).values)].mean(0).cpu() for c in sorted(meta_eval["case"].unique())}


def load_faithful(faithful_dir, L, H, D):
    W, b = torch.zeros(L, H, D), torch.zeros(L, H)
    for l in range(L):
        for h in range(H):
            sd = torch.load(os.path.join(faithful_dir, f"linear_probe_{l}_{h}.pth"), map_location="cpu")
            W[l, h], b[l, h] = sd["linear.weight"].squeeze(), sd["linear.bias"].squeeze()
    acc = pickle.load(open(os.path.join(faithful_dir, "linear_accuracies_dict_mha.pkl"), "rb"))
    return W, b, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--faithful_dir", default=None, help="Folder with the faithful linear_probe_*.pth and accuracy pkl")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta, acts = load_acts(args.acts_dir, which=("mha",))
    X = acts["mha"]
    n, L, H, D = X.shape
    print(f"{n} dialogues, {L} layers x {H} heads x {D} dims")
    y_syco = torch.tensor(meta["syco"].values)

    # Authors' subset in their order, and their per-dialogue split.
    auth = meta[meta.authors_pos >= 0].sort_values("authors_pos")
    auth_idx = auth.index.values
    tr_pos, va_pos = train_test_split(np.arange(len(auth_idx)), test_size=0.2, random_state=SEED)

    # Question-level split.
    qids = np.sort(meta.qid.unique())
    q_tr, q_va = train_test_split(qids, test_size=0.2, random_state=SEED)
    is_tr = meta.qid.isin(q_tr).values
    is_va = ~is_tr
    auth_tr = auth_idx[np.isin(meta.loc[auth_idx, "qid"], q_tr)]
    auth_va = auth_idx[np.isin(meta.loc[auth_idx, "qid"], q_va)]
    all_tr, all_va = np.where(is_tr)[0], np.where(is_va)[0]
    meta_va = meta.iloc[all_va]
    print(f"Question split: {len(q_tr)} train / {len(q_va)} held-out questions")

    def T(idx):
        return torch.as_tensor(idx, dtype=torch.long)

    sets = {}
    print("Training A_original (authors' data and per-dialogue split)...")
    W, b, hist = train_heads(X[T(auth_idx[tr_pos])], y_syco[T(auth_idx[tr_pos])], X[T(auth_idx[va_pos])], y_syco[T(auth_idx[va_pos])],
                             args.epochs, args.batch, args.lr, device=args.device)
    sets["A_original"] = (W, b, hist, "acc")
    print("Training A_grouped (authors' data, split by question)...")
    W, b, hist = train_heads(X[T(auth_tr)], y_syco[T(auth_tr)], X[T(auth_va)], y_syco[T(auth_va)],
                             args.epochs, args.batch, args.lr, device=args.device)
    sets["A_grouped"] = (W, b, hist, "acc")
    print("Training B (all four cases, split by question, class-balanced)...")
    pw = float((y_syco[T(all_tr)] == 0).sum() / (y_syco[T(all_tr)] == 1).sum())
    W, b, hist = train_heads(X[T(all_tr)], y_syco[T(all_tr)], X[T(all_va)], y_syco[T(all_va)],
                             args.epochs, args.batch, args.lr, pos_weight=pw, device=args.device)
    sets["B"] = (W, b, hist, "bal_acc")

    faithful_acc = None
    if args.faithful_dir:
        Wf, bf, faithful_acc = load_faithful(args.faithful_dir, L, H, D)
        sets["faithful"] = (Wf, bf, None, None)

    # Per-head table.
    X_va = X[T(all_va)]
    rows = []
    for name, (W, b, hist, metric) in sets.items():
        torch.save({"W": W, "b": b}, os.path.join(args.out_dir, f"probes_{name}.pt"))
        pred = predict(X_va.to(args.device), W, b)
        rates = case_rates(pred, meta_va)
        y_va = y_syco[T(all_va)].to(pred.device).float()
        bal_all = balanced_acc(pred, y_va).cpu()
        for l in range(L):
            for h in range(H):
                r = {"set": name, "head_id": f"{l}_{h}", "layer": l, "headidx": h, "bal_acc_all_cases": float(bal_all[l, h])}
                if hist is not None:
                    r["val_best"] = float(hist[metric][:, l, h].max())
                    r["val_last"] = float(hist[metric][-1, l, h])
                elif faithful_acc is not None:
                    r["val_best"] = faithful_acc[f"{l}_{h}"] / 100
                for c, v in rates.items():
                    r[c] = float(v[l, h])
                rows.append(r)
    df = pd.DataFrame(rows)
    cases = sorted(meta.case.unique())
    syco_cases = sorted(meta[meta.syco == 1].case.unique())
    ctrl_cases = sorted(meta[meta.falsehood_control == 1].case.unique())
    df["hit_rate"] = df[syco_cases].mean(1)  # sycophantic dialogues called sycophantic
    df["falsehood_ctrl_rate"] = df[ctrl_cases].mean(1)  # false-ending, non-sycophantic dialogues called sycophantic
    df["syco_vs_ctrl_acc"] = 0.5 * (df.hit_rate + 1 - df.falsehood_ctrl_rate)  # ~0.5 for a pure falsehood detector
    df.to_csv(os.path.join(args.out_dir, "per_head.csv"), index=False)

    # Summary: top-k heads by each set's own selection metric, plus the faithful run's top-k heads.
    faithful_top = sorted(faithful_acc, key=faithful_acc.get, reverse=True)[:args.k] if faithful_acc else None
    summ = []
    for name in sets:
        d = df[df.set == name]
        selections = {"own top-k": d.nlargest(args.k, "val_best")["head_id"].tolist()}
        if faithful_top:
            selections["faithful top-k"] = faithful_top
        for sel_name, heads in selections.items():
            s = d[d["head_id"].isin(heads)]
            summ.append({"set": name, "heads": sel_name,
                         **{c: s[c].mean() for c in ["val_best", "val_last", "hit_rate", "falsehood_ctrl_rate", "syco_vs_ctrl_acc", "bal_acc_all_cases"] + cases if c in s}})
    summary = pd.DataFrame(summ)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print("\nMean over the selected heads (rates = fraction called sycophantic on held-out questions):")
        print(summary.to_string(index=False))

    # Agreement of A_original with the faithful run (same data and split, so it should be high).
    if faithful_acc:
        a = df[df.set == "A_original"].set_index("head_id").val_best
        f = pd.Series(faithful_acc) / 100
        top_a = set(a.nlargest(args.k).index)
        info = {"spearman_A_original_vs_faithful": float(a.rank().corr(f.reindex(a.index).rank())),
                "topk_overlap": len(top_a & set(faithful_top))}
        json.dump(info, open(os.path.join(args.out_dir, "sanity.json"), "w"), indent=1)
        print("\nSanity check vs the faithful run:", info)

    heatmaps(df, L, H, args.out_dir)


def heatmaps(df, L, H, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [s for s in ["A_grouped", "B", "faithful"] if s in df.set.unique()]
    cols = [("val_best", "Validation accuracy (own task)"), ("falsehood_ctrl_rate", "False-ending controls called sycophantic")]
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 3.2 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        d = df[df.set == name]
        for j, (col, title) in enumerate(cols):
            m = np.full((L, H), np.nan)
            m[d.layer, d.headidx] = d[col]
            im = axes[i, j].imshow(m.T, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1)
            axes[i, j].set_title(f"{name}: {title}", fontsize=9)
            axes[i, j].set_xlabel("Layer")
            axes[i, j].set_ylabel("Head")
            fig.colorbar(im, ax=axes[i, j], fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmaps.png"), dpi=150)


if __name__ == "__main__":
    main()
