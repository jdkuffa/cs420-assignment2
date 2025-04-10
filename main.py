import re
import torch
import sacrebleu
import pandas as pd

from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from datasets import load_dataset

# ------------------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------------------
# Load dataset from CSV files
#dataset = load_dataset('csv', data_files={'train': 'ft_train.csv', 'valid': 'ft_valid.csv', 'test': 'ft_test.csv'})
dataset = load_dataset('csv', data_files={'train': 'ft_train_subset.csv', 'valid': 'ft_valid_subset.csv', 'test': 'ft_test_subset.csv'})

# Convert DataFrame datasets to Hugging Face Dataset
train_dataset = dataset['train']
valid_dataset = dataset['valid']
test_dataset = dataset['test']


# ------------------------------------------------------------------------
# 2. Load Model and Tokenizer
# ------------------------------------------------------------------------
# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model from Hugging Face using the checkpoint name
model_checkpoint = "Salesforce/codet5-small"
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

# Move model to the device
model.to(device)

# Load pre-trained tokenizer from Hugging Face and add custom token
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens(["<MASK>"]) # Imagine we need an extra token - this line adds the extra token to the vocabulary

# Resize model's embedding layer to accommodate new vocabulary size
model.resize_token_embeddings(len(tokenizer))


# ------------------------------------------------------------------------
# 2. Modify Dataset by Masking and Flattening
# ------------------------------------------------------------------------
def flatten(text):
    return text.replace("\n", " ").replace(" ", " ").replace("__NEW_LINE__", " ").strip()

def normalize(text):
    text = text.replace("__NEW_LINE__", " ")  # if used
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", "", text)  # remove all whitespace
    return text

def match_target(target):
  tokens = target.strip().split()
  return r"\s*".join(map(re.escape, tokens))

def mask_method(method, target_block):
    norm_method = normalize(method)
    norm_target_block = normalize(target_block)

    if norm_target_block in norm_method:
        pattern = match_target(target_block)
        try:
            masked_method = re.sub(pattern, "<MASK>", method, count=1)
            return masked_method
        except Exception as e:
            return method
    else:
        return method

def flatten_and_mask(examples):
    flattened_and_masked_methods = []

    for i in range(len(examples['cleaned_method'])):

        # Mask method
        target_block = examples['target_block'][i]
        masked_method = mask_method(examples['cleaned_method'][i], target_block)

        # Flatten method
        flatten_masked_method = flatten(masked_method)

        flattened_and_masked_methods.append(flatten_masked_method)

    examples['flattened_and_masked_method'] = flattened_and_masked_methods
    return examples

dataset = dataset.map(flatten_and_mask, batched=True, num_proc=4)


# ------------------------------------------------------------------------
# 3. Fine-Tune Model Using Tokenizer
# ------------------------------------------------------------------------
def preprocess_function(examples):
    inputs = examples["flattened_and_masked_method"]
    targets = examples["target_block"]
    # print("Inputs:", inputs[0])
    # print("Targets:", targets[0])
    model_inputs = tokenizer(
        inputs, 
        max_length=256, 
        truncation=True, 
        padding="max_length"
    )
    labels = tokenizer(
        targets, 
        max_length=256, 
        truncation=True, 
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True)


# ------------------------------------------------------------------------
# 4. Define Training Arguments and Trainer
# ------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./codet5-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=100,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


# ------------------------------------------------------------------------
# 5. Train the Model
# ------------------------------------------------------------------------
trainer.train()


# ------------------------------------------------------------------------
# 6. Evaluate on Test Set
# ------------------------------------------------------------------------
metrics = trainer.evaluate(tokenized_datasets["test"])
print("Test Evaluation Metrics: ", metrics)


# ------------------------------------------------------------------------
# 7. Evaluate on Test Set
# ------------------------------------------------------------------------
def bleu_score(predictions, references):
    formatted_references = [[ref] for ref in references]
    result = sacrebleu.corpus_bleu(predictions, formatted_references, smooth_method="exp")
    return result.score / 100

def exact_match_score(predictions, references):
    return sum(p.strip() == r.strip() for p, r in zip(predictions, references)) / len(predictions)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.DataFrame(columns=["input", "expected_if", "predicted_if", "code_bleu_score", "bleu_4_score", "exact_match"])

for i in range(len(tokenized_datasets["test"])):
    input_text = tokenized_datasets["test"][i]["flattened_and_masked_method"]
    expected_if = tokenized_datasets["test"][i]["target_block"]

    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, padding="max_length", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.generate(**inputs, max_length=256)
    predicted_if = tokenizer.decode(output[0], skip_special_tokens=True)

    code_bleu_score = sacrebleu.sentence_bleu(predicted_if, [expected_if]).score / 100
    bleu_4_score = bleu_score([predicted_if], [expected_if])

    exact_match = exact_match_score([predicted_if], [expected_if])

    df.loc[len(df)] = [input_text, expected_if, predicted_if, code_bleu_score, bleu_4_score, exact_match]

df.to_csv("testset-results.csv", index=False)

