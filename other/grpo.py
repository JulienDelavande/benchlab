# 1) Charger le modèle Qwen2.5 VL
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

from datasets import load_dataset
from trl import GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]

trainer = GRPOTrainer(
   # model="Qwen/Qwen2-0.5B-Instruct",
    model=model_name,
    reward_funcs=reward_num_unique_chars,
    train_dataset=dataset,
)
trainer.train()