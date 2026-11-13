"""Step 1-2: build all dialogues and extract last-token activations in one forward pass each.

Saves, in --out_dir:
  meta.csv                 one row per dialogue (question, template, case, labels, authors' position)
  acts_XXXXX.pt            chunks with 'mha' [n, L, H, D] (o_proj input, as in the authors' code)
                           and 'resid' [n, L, hidden] (decoder-layer output), both float16
  template_example.txt     one rendered dialogue (shows the chat template, including its date line)
Re-running skips finished chunks, so an interrupted run resumes.
"""
import argparse
import os

import pandas as pd
import torch
from tqdm.auto import tqdm

import common


class LastTokenRecorder:
    """Records the last position of the o_proj input (per head) and of each decoder layer's output."""

    def __init__(self, model):
        self.L, self.H, self.D, self.hidden = common.model_dims(model)
        self.mha = [None] * self.L
        self.resid = [None] * self.L
        self.handles = []
        for i, layer in enumerate(common.decoder_layers(model)):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._pre(i)))
            self.handles.append(layer.register_forward_hook(self._post(i)))

    def _pre(self, i):
        def hook(module, args):
            self.mha[i] = args[0][0, -1].detach().float().view(self.H, self.D).cpu()
        return hook

    def _post(self, i):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self.resid[i] = h[0, -1].detach().float().cpu()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def tokenize_like_authors(processor, chats):
    # Same two steps as the authors' train.py (this adds a second BOS token, as theirs does).
    templated = processor.apply_chat_template(chats, add_generation_prompt=False, tokenize=False)
    return templated, [processor(text=t, return_tensors="pt")["input_ids"].squeeze() for t in tqdm(templated, desc="Tokenizing")]


def extract(model, processor, chats, meta, out_dir, chunk=500, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(meta).to_csv(os.path.join(out_dir, "meta.csv"), index_label="idx")
    templated, tokens = tokenize_like_authors(processor, chats)
    with open(os.path.join(out_dir, "template_example.txt"), "w") as f:
        f.write(templated[0])

    model.eval()
    rec = LastTokenRecorder(model)
    try:
        for start in range(0, len(tokens), chunk):
            path = os.path.join(out_dir, f"acts_{start:05d}.pt")
            if os.path.exists(path):
                continue
            mha, resid = [], []
            for ids in tqdm(tokens[start:start + chunk], desc=f"Chunk {start // chunk + 1}/{-(-len(tokens) // chunk)}"):
                with torch.no_grad():
                    model(input_ids=ids.unsqueeze(0).to(device))
                mha.append(torch.stack(rec.mha))
                resid.append(torch.stack(rec.resid))
            mha, resid = torch.stack(mha), torch.stack(resid)
            if mha.abs().max() > 6e4 or resid.abs().max() > 6e4:
                raise ValueError("Activation too large for float16; save in float32 instead.")
            torch.save({"start": start, "mha": mha.half(), "resid": resid.half()}, path + ".tmp")
            os.replace(path + ".tmp", path)  # only complete chunks get the final name
    finally:
        rec.remove()


def load_acts(out_dir, which=("mha", "resid")):
    meta = pd.read_csv(os.path.join(out_dir, "meta.csv"), index_col="idx")
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("acts_") and f.endswith(".pt"))
    parts = [torch.load(os.path.join(out_dir, f)) for f in files]
    acts = {k: torch.cat([p[k] for p in parts]) for k in which}
    n = next(iter(acts.values())).shape[0]
    assert n == len(meta), f"{n} activations for {len(meta)} dialogues: extraction incomplete"
    return meta, acts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="llama-3.2")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from utils import load_model

    ds_train = common.load_train_split()
    chats, meta = common.build_dialogues(ds_train)
    n_auth = common.check_against_authors(ds_train, chats, meta)
    print(f"{len(chats)} dialogues for {len(ds_train)} questions; authors' subset reproduced exactly ({n_auth}).")
    model, processor = load_model(args.model_id, device=args.device)
    extract(model, processor, chats, meta, args.out_dir, chunk=args.chunk, device=args.device)
    print("Done:", args.out_dir)
