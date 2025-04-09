from transformers import (T5ForConditionalGeneration,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback)
from datasets import Dataset, DatasetDict, load_dataset
import codebleu
from codebleu import calc_codebleu
# from codebleu.utils import get_tree_sitter_language, PYTHON_LANGUAGE
import pandas as pd
import numpy as np
import re
import evaluate
import sacrebleu
import tabulate
import tree_sitter_python
import pandas as pd
import subprocess
import re
import sacrebleu


# ------------------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------------------

# Load dataset from CSV files
dataset = load_dataset('csv', data_files={'train': 'ft_train_subset.csv', 'valid': 'ft_valid_subset.csv', 'test': 'ft_test_subset.csv'})

# Convert DataFrame datasets to Hugging Face Dataset
train_dataset = dataset['train']
valid_dataset = dataset['valid']
test_dataset = dataset['test']

# # DEMO: Randomly sample 5000 training set & 1000 for validation and 500 for test set
# train_dataset = train_dataset.shuffle(seed=42).select(range(80))
# valid_dataset = valid_dataset.shuffle(seed=42).select(range(10))
# test_dataset = test_dataset.shuffle(seed=42).select(range(10))

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
    num_train_epochs=1, # Change for testing
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
# 7. Take Evaluation Metrics and Save Results to CSV
# ------------------------------------------------------------------------
# Define the working directory for CodeBLEU calculation
working_dir = "/content/CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU/"

def bleu_score(predictions, references):
    formatted_references = [[ref] for ref in references]
    result = sacrebleu.corpus_bleu(predictions, formatted_references, smooth_method="exp")
    return result.score / 100

# Create DataFrame to store results
df = pd.DataFrame(columns=["input", "expected_if", "predicted_if", "code_bleu_score", "bleu_4_score"])

# Define the command and arguments for CodeBLEU calculation
command = [
    "python",
    "calc_code_bleu.py",
    "--refs", "/content/expected_if.txt",
    "--hyp", "/content/predicted_if.txt",
    "--lang", "python",
    "--params", "0.25,0.25,0.25,0.25"
]

# Populate with results (limit to first 5 for testing)
for i in range(5):  # Adjust based on your actual dataset size
    input_text = tokenized_datasets["test"][i]["flattened_and_masked_method"]
    expected_if = tokenized_datasets["test"][i]["target_block"]

    # Decode the predicted output from the model
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=256)
    token_ids = output[0]
    predicted_if = tokenizer.decode(token_ids, skip_special_tokens=True)

    # Save predicted_if and expected_if to text files for BLEU calculation
    with open('/content/predicted_if.txt', 'w') as f:
        f.write(predicted_if)

    with open('/content/expected_if.txt', 'w') as f:
        f.write(expected_if)

    # Run the CodeBLEU calculation command
    result = subprocess.run(command, cwd=working_dir, capture_output=True, text=True)

    # Extract CodeBLEU score from the result using regex
    pattern = r'CodeBLEU\s+score:\s*([0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, result.stdout)

    if matches:
        code_bleu_score = float(matches[0])  # CodeBLEU score
    else:
        code_bleu_score = 0.0  # Default to 0 if no match found

    # Calculate BLEU-4 score using sacrebleu
    bleu_4_score = bleu_score([predicted_if], [expected_if])

    # Save the results to the DataFrame
    df.loc[len(df)] = [input_text, expected_if, predicted_if, code_bleu_score, bleu_4_score]

# Save the results to a CSV
df.to_csv("/content/testset-results.csv", index=False)

# Optionally print the DataFrame to see the results
print(df)
