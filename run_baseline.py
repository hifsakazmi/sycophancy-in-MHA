# run_baseline.py (paper-era branch, commit c21f35d): unsteered two-turn generation.
# Same as faithful-run's version, but this commit's utils.py takes no `device` argument.
import argparse
import os

import pandas as pd
from tqdm.auto import tqdm

from utils import load_model, load_test_data, generate_and_decode_new_tokens

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default="llama-3.2", choices=["llama-3.2", "gemma-3", "qwen-3"])
parser.add_argument("--dataset_id", default="truthfulqa")
parser.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke test).")
parser.add_argument("--out_dir", default="predictions")
args = parser.parse_args()

model, processor = load_model(args.model_id)
model.eval()

questions, answers = load_test_data(args.dataset_id)
questions, answers = list(questions), list(answers)
if args.limit:
    questions, answers = questions[: args.limit], answers[: args.limit]

initial_answer, final_answer = [], []
for question in tqdm(questions):
    res_1, res_2 = generate_and_decode_new_tokens(question, model, processor, args.model_id)
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
