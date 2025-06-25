from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DefaultDataCollator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

dataset = load_dataset("csv", data_files="fake_review_old.csv")["train"]
dataset = dataset.rename_column("label", "labels")
dataset = dataset.rename_column("text", "input")


def preprocess_labels(example):
    if isinstance(example['labels'], str):
        example['labels'] = 1 if example['labels'].upper() == "REAL" else 0
    return example

dataset = dataset.map(preprocess_labels)


def format_prompt(example):
    prompt = f"Review:\n{example['input']}\n\nIs this review FAKE or REAL?"
    example["prompt"] = prompt
    return example

dataset = dataset.map(format_prompt)


model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    tokenized = tokenizer(
        batch["prompt"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    tokenized["labels"] = [int(label) for label in batch["labels"]]  # batch-safe
    return tokenized


dataset = dataset.map(tokenize, batched=True)


split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


data_collator = DefaultDataCollator()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(labels, predictions, output_dict=True)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }

output_dir = "output"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-4,
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
    load_best_model_at_end=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


trainer.train()


adapter_path = f"{output_dir}/adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(f"{output_dir}/tokenizer")

# Save merged model
merged_path = f"{output_dir}/merged_model"
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_path)

print(f"Adapter saved to: {adapter_path}")
print(f"Tokenizer saved to: {output_dir}/tokenizer")
print(f" Merged model saved to: {merged_path}")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator
from sklearn.metrics import classification_report

# Clean up after training
# del trainer
torch.cuda.empty_cache()

eval_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=default_data_collator)

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=["FAKE", "REAL"]))