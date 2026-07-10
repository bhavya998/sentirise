"""Quick dataset check."""
from sentirise.dataset import load_imdb_dataset

train, eval_ds = load_imdb_dataset(max_train=10, max_eval=5)
print(f"Train: {len(train)} samples")
print(f"Eval: {len(eval_ds)} samples")
sample = train[0]["text"]
print(f"Example ({len(sample)} chars):")
print(sample[:300])
