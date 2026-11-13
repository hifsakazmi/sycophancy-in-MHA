"""Write a probe set trained by probes.py (e.g. probes_B.pt) in the authors' probe-file format,
so their unchanged inference_mha.py can steer with it.

Uses the faithful run's accuracy file, so the SAME top-k heads are steered and only the direction changes.
Sigma = std of all dialogues' activations projected on each probe direction (the authors' recipe).
"""
import argparse
import os

import torch

from dim import write_head_dir
from extract import load_acts

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes_file", required=True, help="e.g. …/falsehood/probes/probes_B.pt")
    ap.add_argument("--acts_dir", required=True)
    ap.add_argument("--faithful_dir", required=True)
    ap.add_argument("--out_dir", required=True, help="e.g. …/falsehood/dim/heads_probeB")
    ap.add_argument("--model_id", default="llama-3.2")
    args = ap.parse_args()

    W = torch.load(args.probes_file)["W"].float()
    _, acts = load_acts(args.acts_dir, which=("mha",))
    d = write_head_dir(W, acts["mha"], args.faithful_dir, args.out_dir, args.model_id)
    print("Wrote", d)
