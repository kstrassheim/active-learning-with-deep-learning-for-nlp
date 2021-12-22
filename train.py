import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import EarlyStoppingCallback


# Read data
data = pd.read_json('data/dataset_1.json')
data["label_bin"] = np.where(data["label"] > 1, 1, 0)
data["label_n"] = data["label"] - 1
data["display_text"] = [d[1]['text'][d[1]['displayTextRangeStart']: d[1]['getDisplayTextRangeEnd']] for d in data[["text","displayTextRangeStart", "getDisplayTextRangeEnd"]].iterrows()]
max_display_text_length = len(data.iloc[np.argmax(data['display_text'].to_numpy())]['display_text'])

# X = data.display_text.to_list()
# y = data.label.to_list()

validation_split_ratio = 0.2
random_seed = 1337

# Define pretrained tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=4)

# ----- 1. Preprocess data -----#
# Preprocess data
X = list(data["display_text"])
y = list(data["label_n"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split_ratio, random_state=random_seed, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split_ratio * len(X) / len(X_train), random_state=random_seed, shuffle=True)

X_train_trans = tokenizer(X, truncation=True, padding=True, max_length=512)
X_test_trans = tokenizer(X_test, truncation=True, padding=True, max_length=512)
X_val_trans = tokenizer(X_val, truncation=True, padding=True, max_length=512)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

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

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    seed=0,
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train pre-trained model
trainer.train()

# ----- 3. Predict -----#
# Load test data
# test_data = pd.read_csv("test.csv")
# X_test = list(test_data["review"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# # Load trained model
# model_path = "./output/checkpoint-50000"
# model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Make prediction
raw_pred, _, _ = trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)