"""Step 6: steer the residual stream at one layer with a DiM direction and answer the 164 test questions.

Adds scale * sigma * direction to the output of decoder layer `layer`, on generated tokens only (the prompt
and the first answer token are not steered), matching how the authors' pyvene setup steers heads.
Writes a CSV with the same columns as the authors' inference scripts, so judge.py works on it unchanged.
"""
import argparse
import os

import pandas as pd
import torch
from tqdm.auto import tqdm

import common


class ResidualSteer:
    def __init__(self, model, layer, vector):
        self.vector = vector
        self.handle = common.decoder_layers(model)[layer].register_forward_hook(self.hook)

    def hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] != 1:  # prompt pass (with KV cache, generated tokens arrive one at a time)
            return None
        h = h + self.vector.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def remove(self):
        self.handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim_file", required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", default="llama-3.2")
    ap.add_argument("--dataset_id", default="truthfulqa")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from utils import load_model, load_test_data, generate_and_decode_new_tokens

    d = torch.load(args.dim_file)
    model, processor = load_model(args.model_id, device=args.device)
    model.eval()
    vec = (args.scale * d["sigma"] * d["direction"]).to(args.device)
    steer = ResidualSteer(model, d["layer"], vec)
    print(f"Steering layer {d['layer']} with scale {args.scale} (|vector| = {vec.norm():.3f})")

    questions, answers = load_test_data(args.dataset_id)
    questions, answers = list(questions), list(answers)
    if args.limit:
        questions, answers = questions[:args.limit], answers[:args.limit]
    first, second = [], []
    for q in tqdm(questions):
        r1, r2 = generate_and_decode_new_tokens(q, model, processor, args.model_id, device=args.device)
        first.append(r1)
        second.append(r2)
    steer.remove()

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_first{args.limit}" if args.limit else ""
    tag = os.path.splitext(os.path.basename(args.dim_file))[0].replace("resid_", "")  # e.g. dimA, dimC
    path = os.path.join(args.out_dir, f"{args.dataset_id}_{args.model_id}_answers_resid{d['layer']}_{args.scale}_{tag}{suffix}.csv")
    pd.DataFrame({"question": questions, "correct_answer": answers, "initial_answer": first, "final_answer": second}).to_csv(path)
    print("Saved", path)


if __name__ == "__main__":
    main()
