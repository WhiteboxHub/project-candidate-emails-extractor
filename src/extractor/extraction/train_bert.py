import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

from pathlib import Path

# Fix pathing to be relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2] # src/extractor/extraction -> src/extractor -> src -> root

# Configuration
DATA_PATH = PROJECT_ROOT / "data" / "job_training_data.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = PROJECT_ROOT / "models" / "bert_binary_classifier"
LOGS_DIR = PROJECT_ROOT / "logs" / "training"

# Ensure directories exist
OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Label Mapping
LABEL_MAP = {
    "valid_job": 0,
    "junk": 1
}
ID2LABEL = {0: "valid job requirement", 1: "junk text or spam"}
LABEL2ID = {"valid job requirement": 0, "junk text or spam": 1}

def train():
    print(f"Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df)} records.")
    
    # Map labels to IDs
    df['label'] = df['label'].map(LABEL_MAP)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.1, 
        random_state=42
    )

    # Initialize Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Tokenize
    def tokenize_function(texts):
        return tokenizer(texts, truncation=True, padding=True, max_length=512)

    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)

    # Create Dataset objects
    class JobDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = JobDataset(train_encodings, train_labels)
    val_dataset = JobDataset(val_encodings, val_labels)

    # Load Model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=LOGS_DIR,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete.")

if __name__ == "__main__":
    train()
