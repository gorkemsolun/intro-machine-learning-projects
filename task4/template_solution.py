# Task 4 - Text classification (sentiment of product reviews).
#
# Goal: given a review's "title" and "sentence", predict whether the review is positive (1) or negative (0). Submissions are scored by accuracy.
#
# Approach: fine-tune a pretrained DistilBERT transformer for binary sequence classification. The handout notes that frozen weights already suffice to pass the hard baseline, but fine-tuning is encouraged for those with the compute ("This best mimics real world use cases of pretrained transformers."). On an Apple GPU (MPS) one epoch over the 12.5k reviews takes ~1.5 min, so a few epochs are cheap and lift accuracy from ~86% (frozen features) to ~92%.
#
#   1. Combine each review's "title" and "sentence" into a single text.
#   2. Tokenize with the DistilBERT tokenizer (dynamic per-batch padding).
#   3. Fine-tune DistilBertForSequenceClassification with AdamW + a linear
#      warmup/decay schedule, selecting the epoch with the best held-out
#      validation accuracy (early stopping) to guard against overfitting.
#   4. Predict on the test set and write one label (0/1) per line to result.txt.

import os
# We disable low-level log outputs by default to keep the terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Resolve all paths relative to this script so it runs from any working dir.
HERE = os.path.dirname(os.path.abspath(__file__))
def path(name):
    return os.path.join(HERE, name)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256 # covers ~p99 of the review lengths in train.csv
BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
VALIDATION_FRACTION = 0.1

train_values = pd.read_csv(path("train.csv"))
test_values = pd.read_csv(path("test_no_score.csv"))


def build_text(df):
    """Combine the (possibly missing) title and sentence into one string."""
    title = df["title"].fillna("").astype(str)
    sentence = df["sentence"].fillna("").astype(str)
    return (title + ". " + sentence).str.strip().tolist()


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Dataset: holds raw review texts (and optional labels). Tokenization happens in the collate function so we can pad dynamically per-batch (much faster than padding everything to MAX_LENGTH).
class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        if self.labels is None:
            return self.texts[index]
        return self.texts[index], self.labels[index]


def make_collate(has_labels):
    """Tokenize a batch -> {input_ids, attention_mask[, labels]} with dynamic padding."""
    def collate(batch):
        if has_labels:
            texts, labels = zip(*batch)
        else:
            texts, labels = batch, None
        encodings = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        item = {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]}
        if labels is not None:
            item["labels"] = torch.tensor(labels, dtype=torch.long)
        return item
    return collate


# Build train / validation split (stratification is unnecessary as classes are already balanced) and the test set.
all_texts = build_text(train_values)
all_labels = train_values["score"].astype(int).tolist()
test_texts = build_text(test_values)

permutation = np.random.default_rng(SEED).permutation(len(all_texts))
n_values = int(len(permutation) * VALIDATION_FRACTION)
validation_index, train_index = permutation[:n_values].tolist(), permutation[n_values:].tolist()

train_dataset = SentimentDataset([all_texts[i] for i in train_index], [all_labels[i] for i in train_index])
validation_dataset = SentimentDataset([all_texts[i] for i in validation_index], [all_labels[i] for i in validation_index])
test_dataset = SentimentDataset(test_texts)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, collate_fn=make_collate(True))
validation_loader = DataLoader(validation_dataset, batch_size=2 * BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=make_collate(True))
test_loader = DataLoader(test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False,
                         num_workers=0, collate_fn=make_collate(False))


# Classifier: DistilBERT with a sequence classification head, fine-tuned end to end. forward() returns the class logits.
class SentimentClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=2):
        super().__init__()
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).logits


model = SentimentClassifier().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps,
)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        preds = model(**batch).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


best_validation_accuracy = -1.0
best_state = None
for epoch in range(NUM_EPOCHS):
    model.train()
    running = 0.0
    for batch in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        logits = model(**batch)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        running += loss.item()

    validation_accuracy = evaluate(validation_loader)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  train_loss={running / len(train_loader):.4f}  val_acc={validation_accuracy:.4f}")
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# Restore the best performing checkpoint before predicting on the test set.
if best_state is not None:
    model.load_state_dict(best_state)
print(f"Best validation accuracy: {best_validation_accuracy:.4f}")

# Predict on the test set and write the submission (one label per line).
model.eval()
with torch.no_grad():
    results = []
    for batch in tqdm(test_loader, total=len(test_loader), desc="Predicting test"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        predictions = model(**batch).argmax(dim=1)
        results.append(predictions.cpu().numpy())

    output_path = path("result.txt")
    with open(output_path, "w") as f:
        for val in np.concatenate(results):
            f.write(f"{int(val)}\n")
print(f"Wrote {sum(len(r) for r in results)} predictions to {output_path}")
