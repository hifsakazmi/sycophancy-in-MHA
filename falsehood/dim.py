"""Step 5-6: difference-in-means (DiM) steering directions.

Head DiM, written in the authors' probe-file format so their unchanged inference_mha.py can steer with it:
  <out_dir>/heads_dimA/trained_probe_sycophancy/<model>/linear_probe_{l}_{h}.pth   (direction as 'linear.weight')
                                                        linear_std_mha_{l}_{h}.pt    (sigma, the authors' way)
                                                        linear_accuracies_dict_mha.pkl (copied from the faithful
                                                        run, so the SAME top-k heads are steered)
  dimA = mean(sycophantic) - mean(not sycophantic) on the authors' 2,612 dialogues and labels
  dimB = the same on all four cases (adds the false-ending, non-sycophantic dialogues to the negative class)

Residual DiM (Baez et al. style): one layer, chosen by held-out separation (Cohen's d) among the middle third
of layers; saved as <out_dir>/resid_dimA.pt for steer_residual.py.

Also writes direction_cosines.csv: for the faithful top-k heads, the cosine between the faithful probe, dimA,
dimB and a "falsehood direction" (mean of false-ending minus true-ending dialogues).
"""
import argparse
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from extract import load_acts

SEED = 3407


def dim(X, y):
    X = X.float()
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def unit(v):
    return v / v.norm(dim=-1, keepdim=True)


def proj_std(X, v):
    """Std of activations projected on the unit direction (per head if v is [L,H,D])."""
    return torch.einsum("n...d,...d->n...", X.float(), unit(v)).std(0)


def cos(a, b):
    return float((unit(a) * unit(b)).sum())


def write_head_dir(v, X_sigma, faithful_dir, dst, model_id):
    d = os.path.join(dst, "trained_probe_sycophancy", model_id)
    os.makedirs(d, exist_ok=True)
    sig = proj_std(X_sigma, v)
    L, H, _ = v.shape
    for l in range(L):
        for h in range(H):
            torch.save({"linear.weight": v[l, h][None, :].clone(), "linear.bias": torch.zeros(1)},
                       os.path.join(d, f"linear_probe_{l}_{h}.pth"))
            torch.save(sig[l, h].clone(), os.path.join(d, f"linear_std_mha_{l}_{h}.pt"))
    shutil.copy(os.path.join(faithful_dir, "linear_accuracies_dict_mha.pkl"), d)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True)
    ap.add_argument("--faithful_dir", required=True, help="Faithful run's probe folder (…/trained_probe_sycophancy/llama-3.2)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", default="llama-3.2")
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta, acts = load_acts(args.acts_dir)
    Xh, Xr = acts["mha"], acts["resid"]
    auth = meta[meta.authors_pos >= 0].sort_values("authors_pos")
    ai = torch.as_tensor(auth.index.values)
    y_auth = torch.as_tensor(auth.syco.values)
    y_all = torch.as_tensor(meta.syco.values)
    final_false = torch.as_tensor((meta.final == "F").values.astype(int))

    # ---- head directions
    vA = dim(Xh[ai], y_auth)
    vB = dim(Xh, y_all)
    vF = dim(Xh, final_false)  # falsehood direction
    dA = write_head_dir(vA, Xh[ai], args.faithful_dir, os.path.join(args.out_dir, "heads_dimA"), args.model_id)
    dB = write_head_dir(vB, Xh, args.faithful_dir, os.path.join(args.out_dir, "heads_dimB"), args.model_id)
    print("Head DiM folders:\n ", dA, "\n ", dB)

    acc = pickle.load(open(os.path.join(args.faithful_dir, "linear_accuracies_dict_mha.pkl"), "rb"))
    top = sorted(acc, key=acc.get, reverse=True)[:args.k]
    rows = []
    for hs in top:
        l, h = map(int, hs.split("_"))
        p = torch.load(os.path.join(args.faithful_dir, f"linear_probe_{l}_{h}.pth"), map_location="cpu")["linear.weight"].squeeze().float()
        rows.append({"head": hs, "faithful_acc": acc[hs],
                     "cos(probe, dimA)": cos(p, vA[l, h]), "cos(probe, falsehood)": cos(p, vF[l, h]),
                     "cos(dimA, falsehood)": cos(vA[l, h], vF[l, h]), "cos(dimB, falsehood)": cos(vB[l, h], vF[l, h]),
                     "cos(dimA, dimB)": cos(vA[l, h], vB[l, h])})
    cdf = pd.DataFrame(rows)
    cdf.to_csv(os.path.join(args.out_dir, "direction_cosines.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print("\nCosines for the faithful top-%d heads:\n" % args.k, cdf.to_string(index=False))
        print("\nMean:\n", cdf.drop(columns=["head"]).mean().to_string())

    # ---- residual direction: choose the layer by held-out separation in the middle third
    L = Xr.shape[1]
    qids = np.sort(meta.qid.unique())
    q_tr, q_va = train_test_split(qids, test_size=0.2, random_state=SEED)
    tr = torch.as_tensor(auth.qid.isin(q_tr).values)
    Xa, ya = Xr[ai].float(), y_auth
    d_by_layer = {}
    for l in range(L // 3, 2 * L // 3 + 1):
        v = unit(dim(Xa[tr, l], ya[tr]))
        p = Xa[~tr, l] @ v
        p1, p0 = p[ya[~tr] == 1], p[ya[~tr] == 0]
        d_by_layer[l] = float((p1.mean() - p0.mean()) / torch.sqrt((p1.var() + p0.var()) / 2))
    best = max(d_by_layer, key=d_by_layer.get)
    v = dim(Xa[:, best], ya)
    sigma = float((Xa[:, best] @ unit(v)).std())
    vFr = dim(Xr[:, best], final_false)
    torch.save({"layer": best, "direction": unit(v), "sigma": sigma, "d_by_layer": d_by_layer,
                "cos_with_falsehood": cos(v, vFr)}, os.path.join(args.out_dir, "resid_dimA.pt"))
    print(f"\nResidual layer chosen: {best} (held-out Cohen's d {d_by_layer[best]:.2f}); sigma {sigma:.3f}; "
          f"cos with falsehood direction {cos(v, vFr):.2f}")
    print("Cohen's d by layer:", {k: round(x, 2) for k, x in d_by_layer.items()})


if __name__ == "__main__":
    main()
