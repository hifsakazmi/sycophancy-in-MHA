# run_baseline.py: unsteered two-turn generation on the TruthfulQA test split.
# Put it in the repo root, next to inference_mha.py, and run it from there.
# It reuses the authors' utils.py (same prompts, split, seed, greedy decoding,
# max_new_tokens=128), so its CSV has the same columns as inference_mha.py's.
import argparse
import os

import pandas as pd
import torch
from tqdm.auto import tqdm

from utils import load_model, load_test_data, generate_and_decode_new_tokens

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default="llama-3.2", choices=["llama-3.2", "gemma-3", "qwen-3"])
parser.add_argument("--dataset_id", default="truthfulqa")
parser.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke test).")
parser.add_argument("--out_dir", default="predictions_baseline")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

model, processor = load_model(args.model_id, device=args.device)
model.eval()

questions, answers = load_test_data(args.dataset_id)
questions, answers = list(questions), list(answers)
if args.limit:
    questions, answers = questions[: args.limit], answers[: args.limit]

initial_answer, final_answer = [], []
for question in tqdm(questions):
    res_1, res_2 = generate_and_decode_new_tokens(question, model, processor, args.model_id, device=args.device)
    initial_answer.append(res_1)
    final_answer.append(res_2)

os.makedirs(args.out_dir, exist_ok=True)
suffix = f"_first{args.limit}" if args.limit else ""
path = os.path.join(args.out_dir, f"{args.dataset_id}_{args.model_id}_answers_baseline{suffix}.csv")
pd.DataFrame({
    "question": questions,
    "correct_answer": answers,
    "initial_answer": initial_answer,
    "final_answer": final_answer,
}).to_csv(path)
print(f"Saved {path}")
