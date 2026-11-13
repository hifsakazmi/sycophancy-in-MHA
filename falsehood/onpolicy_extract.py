"""Version C, steps 3-5: label the model's OWN behaviour and build directions at the decision point.

Input: the judged CSV from onpolicy_generate.py + judge.py. Keeps questions the model first answered correctly;
label 1 = caved (second answer judged incorrect), 0 = held (second answer correct).

For each kept question, one forward pass over the exact second-turn prompt (after the pushback, before the reply);
the activations at its last token are the ones that produce the first token of the second answer.

Outputs in --out_dir:
  acts.pt, meta.csv                 decision-point activations (heads and residual stream) and labels
  decodability.csv                  5-fold cross-validated balanced accuracy of a probe per head and per residual layer
  heads_dimC/trained_probe_sycophancy/<model>/   DiM (caved - held) in the authors' probe format, same top-k heads
  resid_dimC.pt                     residual DiM at the layer with the best cross-validated Cohen's d (middle third)
  cosines.csv                       DiM-C vs the scripted falsehood direction and the scripted DiM (if --scripted_acts)
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from dim import dim, unit, cos, write_head_dir
from extract import LastTokenRecorder, load_acts
from probes import train_heads, predict, balanced_acc

SEED = 3407


def cv_balanced_acc(X, y, folds=5, epochs=25, device="cpu"):
    """Mean 5-fold balanced accuracy (last epoch) of a per-location logistic probe. X: [n, L, H, D]."""
    accs = []
    for tr, va in KFold(folds, shuffle=True, random_state=SEED).split(np.arange(len(y))):
        tr, va = torch.as_tensor(tr), torch.as_tensor(va)
        pw = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
        W, b, _ = train_heads(X[tr], y[tr], X[va], y[va], epochs=epochs, pos_weight=pw, device=device)
        accs.append(balanced_acc(predict(X[va].to(device), W, b), y[va].to(device).float()).cpu())
    return torch.stack(accs).mean(0)


def cv_cohens_d(X, y, folds=5):
    """Cross-validated separation along the DiM direction, per layer. X: [n, L, hidden]."""
    ds = []
    for tr, va in KFold(folds, shuffle=True, random_state=SEED).split(np.arange(len(y))):
        tr, va = torch.as_tensor(tr), torch.as_tensor(va)
        v = unit(X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0))  # [L, hidden]
        p = torch.einsum("nlh,lh->nl", X[va], v)
        p1, p0 = p[y[va] == 1], p[y[va] == 0]
        ds.append((p1.mean(0) - p0.mean(0)) / torch.sqrt((p1.var(0) + p0.var(0)) / 2))
    return torch.stack(ds).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--faithful_dir", required=True)
    ap.add_argument("--scripted_acts", default=None, help="Scripted activations folder (for cosines with the falsehood direction)")
    ap.add_argument("--model_id", default="llama-3.2")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.judged_csv)
    keep = df[(df.initial_label == "CORRECT") & df.final_label.isin(["CORRECT", "INCORRECT"])].reset_index(drop=True)
    y = torch.as_tensor((keep.final_label == "INCORRECT").astype(int).values.copy())
    print(f"{len(df)} questions; {len(keep)} first-correct: {int(y.sum())} caved, {int((y == 0).sum())} held "
          f"(on-policy sycophancy rate {100 * y.float().mean():.1f}%)")

    acts_path = os.path.join(args.out_dir, "acts.pt")
    if os.path.exists(acts_path):
        acts = torch.load(acts_path)
    else:
        from utils import load_model
        model, processor = load_model(args.model_id, device=args.device)
        model.eval()
        rec = LastTokenRecorder(model)
        mha, resid = [], []
        for prompt in tqdm(keep.turn2_prompt, desc="Decision-point activations"):
            ids = processor(prompt, return_tensors="pt")["input_ids"].to(args.device)  # as in utils (Llama path)
            with torch.no_grad():
                model(input_ids=ids)
            mha.append(torch.stack(rec.mha))
            resid.append(torch.stack(rec.resid))
        rec.remove()
        acts = {"mha": torch.stack(mha), "resid": torch.stack(resid)}
        torch.save(acts, acts_path)
        keep.assign(caved=y.numpy()).drop(columns=["turn2_prompt"]).to_csv(os.path.join(args.out_dir, "meta.csv"), index=False)
    Xh, Xr = acts["mha"].float(), acts["resid"].float()
    L, H = Xh.shape[1], Xh.shape[2]

    # Can caving be predicted before the reply?
    acc = pickle.load(open(os.path.join(args.faithful_dir, "linear_accuracies_dict_mha.pkl"), "rb"))
    top = sorted(acc, key=acc.get, reverse=True)[:args.k]
    bal_h = cv_balanced_acc(Xh, y, device=args.device)
    bal_r = cv_balanced_acc(Xr[:, :, None, :], y, device=args.device)[:, 0]
    rows = [{"location": f"head {l}_{h}", "layer": l, "bal_acc": float(bal_h[l, h]), "faithful_top": f"{l}_{h}" in top}
            for l in range(L) for h in range(H)]
    rows += [{"location": f"residual {l}", "layer": l, "bal_acc": float(bal_r[l]), "faithful_top": False} for l in range(L)]
    dec = pd.DataFrame(rows)
    dec.to_csv(os.path.join(args.out_dir, "decodability.csv"), index=False)
    print("\n5-fold balanced accuracy for predicting caving from the decision point (0.5 = chance):")
    print(f"  faithful top-{args.k} heads, mean: {dec[dec.faithful_top].bal_acc.mean():.3f}")
    print(f"  best 5 heads: {dec[dec.location.str.startswith('head')].nlargest(5, 'bal_acc')[['location', 'bal_acc']].to_string(index=False, header=False)}")
    print(f"  best residual layer: {dec[dec.location.str.startswith('residual')].nlargest(1, 'bal_acc')[['location', 'bal_acc']].to_string(index=False, header=False)}")

    # Directions
    vC = dim(Xh, y)
    d = write_head_dir(vC, Xh, args.faithful_dir, os.path.join(args.out_dir, "heads_dimC"), args.model_id)
    print("\nHead DiM-C written to", d)
    dcv = cv_cohens_d(Xr, y)
    mid = range(L // 3, 2 * L // 3 + 1)
    best = max(mid, key=lambda l: float(dcv[l]))
    vr = dim(Xr[:, best], y)
    sigma = float((Xr[:, best] @ unit(vr)).std())
    out = {"layer": best, "direction": unit(vr), "sigma": sigma, "d_by_layer": {l: float(dcv[l]) for l in mid}}

    if args.scripted_acts:
        smeta, sacts = load_acts(args.scripted_acts)
        ff = torch.as_tensor((smeta.final == "F").astype(int).values)
        auth = smeta[smeta.authors_pos >= 0].sort_values("authors_pos")
        ai, ya = torch.as_tensor(auth.index.values), torch.as_tensor(auth.syco.values)
        vF, vA = dim(sacts["mha"], ff), dim(sacts["mha"][ai], ya)
        rows = []
        for hs in top:
            l, h = map(int, hs.split("_"))
            rows.append({"head": hs, "cos(dimC, falsehood)": cos(vC[l, h], vF[l, h]), "cos(dimC, scripted dimA)": cos(vC[l, h], vA[l, h])})
        cdf = pd.DataFrame(rows)
        out["cos_with_scripted_falsehood"] = cos(vr, dim(sacts["resid"][:, best], ff))
        out["cos_with_scripted_dimA"] = cos(vr, dim(sacts["resid"][ai, best], ya))
        cdf.to_csv(os.path.join(args.out_dir, "cosines.csv"), index=False)
        print(f"\nCosines on the faithful top-{args.k} heads (mean): "
              f"dimC vs falsehood {cdf['cos(dimC, falsehood)'].mean():.2f}, dimC vs scripted dimA {cdf['cos(dimC, scripted dimA)'].mean():.2f}")
        print(f"Residual layer {best}: dimC vs falsehood {out['cos_with_scripted_falsehood']:.2f}, vs scripted dimA {out['cos_with_scripted_dimA']:.2f}")

    torch.save(out, os.path.join(args.out_dir, "resid_dimC.pt"))
    print(f"Residual DiM-C: layer {best} (CV Cohen's d {float(dcv[best]):.2f}), sigma {sigma:.3f}")
    print("CV Cohen's d by layer:", {l: round(float(dcv[l]), 2) for l in mid})


if __name__ == "__main__":
    main()
