from transformers import (T5ForConditionalGeneration,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback)
from datasets import Dataset, DatasetDict, load_dataset
from codebleu import calc_codebleu
# from codebleu.utils import get_tree_sitter_language, PYTHON_LANGUAGE
import pandas as pd
import numpy as np
import re
import codebleu
import evaluate
import sacrebleu
import tabulate
import tree_sitter_python


# ------------------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------------------

# Load dataset from CSV files
dataset = load_dataset('csv', data_files={'train': 'ft_train.csv', 'valid': 'ft_valid.csv', 'test': 'ft_test.csv'})

# Convert DataFrame datasets to Hugging Face Dataset
train_dataset = dataset['train']
valid_dataset = dataset['valid']
test_dataset = dataset['test']

# DEMO: Randomly sample 5000 training set & 1000 for validation and 500 for test set
train_dataset = train_dataset.shuffle(seed=42).select(range(400))
valid_dataset = valid_dataset.shuffle(seed=42).select(range(100))
test_dataset = test_dataset.shuffle(seed=42).select(range(100))

# Load pre-trained model from Hugging Face using the checkpoint name
model_checkpoint = "Salesforce/codet5-small"
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

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

dataset = dataset.map(flatten_and_mask, batched=True)

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
    num_train_epochs=5, # Change for testing
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
# 7. Take Evaluation Metrics
# ------------------------------------------------------------------------

#Exact Match
#This checks whether the predicted output exactly matches the reference output (character-by-character or token-by-token).
def exact_match_score(predictions, references):
    return sum(p.strip() == r.strip() for p, r in zip(predictions, references)) / len(predictions)

#BLEU Score using SacreBLEU
#This uses Hugging Face’s SacreBLEU metric.

def bleu_score(predictions, references):
    sacrebleu = evaluate.load("sacrebleu")
    formatted_references = [[ref] for ref in references]  # BLEU needs list of list
    results = sacrebleu.compute(predictions=predictions, references=formatted_references)
    return results["score"]

#BLEU Score using CodeBLEU
#This uses a more specialized script from CodeBLEU to compute metric.
#Might require some preprocessing or adjustment depending on formatting.

def codebleu_score(predictions, references, lang="python"):
    # Ensure predictions and references are lists of lists
    if not isinstance(predictions[0], list):
        predictions = [predictions]  # Wrap in a list if not already
    if not isinstance(references[0], list):
        references = [references]  # Wrap in a list if not already

    # Use PYTHON_LANGUAGE directly instead of get_tree_sitter_language
    res = calc_codebleu(references, predictions, lang=PYTHON_LANGUAGE)
    print(res)
    return res

#Example data
predictions = ["if x > 5:", "if y == 10:"]
references = ["if x > 5:", "if y == 10:"]

#Run evaluations
exact_match = exact_match_score(predictions, references)
# bleu = bleu_score(predictions, references)
# codebleu = codebleu_score(predictions, references)

#Print table
table = [
    ["Exact Match", f"{exact_match:.2f}"],
    ["BLEU-4 (SacreBLEU)", f"{bleu:.2f}"],
    ["CodeBLEU", f"{codebleu:.2f}"]
]

print(tabulate(table, headers=["Metric", "Score (%)"], tablefmt="grid"))

# ------------------------------------------------------------------------
# 8. Save Results to CSV
# ------------------------------------------------------------------------

# Create DataFrame to store test set results
df = pd.DataFrame(columns=["input", "expected_if", "predicted_if"])  # "code_bleu", "bleu"

# Populate with results
# for i in range(len(tokenized_datasets["test"])):

for i in range(30):
    # Take cleaned method as input
    input_text = tokenized_datasets["test"][i]["flattened_and_masked_method"]

    # Pull target block as the expected if statement
    expected_if = tokenized_datasets["test"][i]["target_block"]

    # Decode the prediction
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=256)
    token_ids = output[0]
    predicted_if = tokenizer.decode(token_ids, skip_special_tokens=True)

    # Produce evaluation scores
    code_bleu = codebleu_score([predicted_if], [expected_if])
    bleu = bleu_score([predicted_if], [expected_if])
    exact_match = exact_match_score([predicted_if], [expected_if])

    # Save each input's results to DataFrame
    df.loc[i] = [input_text, expected_if, predicted_if]  # code_bleu, bleu

# Convert DataFrame to CSV file
df.to_csv("testset-results.csv", index=False)