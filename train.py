
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

random_seed = 1337
from os import environ
environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# !! important !! import torch after setting cublas deterministic or it will not work !!
import torch
from transformers import TrainingArguments, Trainer, DistilBertTokenizer, DistilBertForSequenceClassification
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.manual_seed(random_seed)
np.random.seed(random_seed)
import random
random.seed(random_seed)

# Read data
data = pd.read_json('data/dataset_1.json')
data["label_train"] = data["label"] - 1
data["display_text"] = [d[1]['text'][d[1]['displayTextRangeStart']: d[1]['getDisplayTextRangeEnd']] for d in data[["text","displayTextRangeStart", "getDisplayTextRangeEnd"]].iterrows()]
max_display_text_length = len(data.iloc[np.argmax(data['display_text'].to_numpy())]['display_text'])

validation_split_ratio = 0.2

# Define pretrained tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=4)

# ----- 1. Preprocess data -----#
# Preprocess data
X = list(data["display_text"])
y = list(data["label_train"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split_ratio, random_state=random_seed, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split_ratio * len(X) / len(X_train), random_state=random_seed, shuffle=True)


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])



# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(pred, labels):
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# create datasets
train_dataset = Dataset(tokenizer(X_train, truncation=True, padding=True, max_length=512), y_train)
val_dataset = Dataset(tokenizer(X_val, truncation=True, padding=True, max_length=512), y_val)
test_dataset = Dataset(tokenizer(X_test, padding=True, truncation=True, max_length=512), y_test)

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="epoch",
    eval_steps=1,
    per_device_train_batch_size=100,
    per_device_eval_batch_size=100,
    num_train_epochs=3,
    seed=random_seed,
    load_best_model_at_end=False
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: compute_metrics(p[0], p[1])
)

# Train pre-trained model
trainer.train()

# Test
metrics = trainer.evaluate(test_dataset, metric_key_prefix="")
#     raw_pred, _, _ = trainer.predict(test_dataset)
#     m = compute_metrics(raw_pred, y_test)
print(metrics)