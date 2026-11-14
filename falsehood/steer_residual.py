"""Steer the residual stream at one layer with a saved direction and answer a test set twice.

Adds scale * sigma * direction to the output of decoder layer `layer`, on generated tokens only (the prompt
and the first answer token are not steered), matching how the authors' pyvene setup steers heads.
--scale 0 gives the unsteered model (verified identical), so it doubles as a resumable baseline run.

Writes a CSV with the authors' columns (question, correct_answer, initial_answer, final_answer), so judge.py
works on it unchanged. Saves every --save_every questions and resumes where it stopped, so a disconnect during
a long run (e.g. 1,000 ARC questions) loses at most a few questions. A run is complete when it has as many
rows as the dataset; the script prints that.
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
    ap.add_argument("--dataset_id", default="truthfulqa", help="truthfulqa, arc_challenge, arc_easy or mmlu (authors' loader)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from utils import load_model, load_test_data, generate_and_decode_new_tokens

    d = torch.load(args.dim_file)
    questions, answers = load_test_data(args.dataset_id)
    questions, answers = list(questions), list(answers)
    if args.limit:
        questions, answers = questions[:args.limit], answers[:args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_first{args.limit}" if args.limit else ""
    tag = os.path.splitext(os.path.basename(args.dim_file))[0].replace("resid_", "")  # e.g. dimA, dimC, random
    path = os.path.join(args.out_dir, f"{args.dataset_id}_{args.model_id}_answers_resid{d['layer']}_{args.scale}_{tag}{suffix}.csv")

    done = 0
    if os.path.exists(path):
        prev = pd.read_csv(path)
        done = len(prev)
        assert list(prev.question) == questions[:done], "Existing file does not match this question list; move it away first."
    print(f"{path}\n{len(questions)} questions, {done} already done")
    if done >= len(questions):
        print("Complete.")
        return

    model, processor = load_model(args.model_id, device=args.device)
    model.eval()
    vec = (args.scale * d["sigma"] * d["direction"]).to(args.device)
    steer = ResidualSteer(model, d["layer"], vec)
    print(f"Steering layer {d['layer']} with scale {args.scale} (|vector| = {vec.norm():.3f})")

    buf = []

    def flush():
        if buf:
            pd.DataFrame(buf).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            buf.clear()

    try:
        for i in tqdm(range(done, len(questions)), initial=done, total=len(questions)):
            r1, r2 = generate_and_decode_new_tokens(questions[i], model, processor, args.model_id, device=args.device)
            buf.append({"question": questions[i], "correct_answer": answers[i], "initial_answer": r1, "final_answer": r2})
            if len(buf) >= args.save_every:
                flush()
    finally:
        flush()
        steer.remove()
    n = len(pd.read_csv(path))
    print(f"Saved {path}: {n}/{len(questions)} questions" + (" (complete)" if n == len(questions) else " (rerun the same command to resume)"))


if __name__ == "__main__":
    main()
