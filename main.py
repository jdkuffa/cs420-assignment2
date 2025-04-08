from datasets import load_dataset
from transformers import (T5ForConditionalGeneration,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback)
import re

dataset = load_dataset('csv', data_files={'train': 'ft_train.csv', 'valid': 'ft_valid.csv', 'test': 'ft_test.csv'})

train_dataset = dataset['train']
eval_dataset = dataset['valid']
test_dataset = dataset['test']

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
    # Replace newlines with spaces and strip leading andtrailing whitespace
    return re.sub(r'\s+', ' ', text).strip() # return text.replace("\n", " ").strip()


def mask_method(method, target_block):
    # Get start and end index of the target block
    start_index = method.find(target_block)
    end_index = start_index + len(target_block)

    # Mask the target block in the method
    masked_method = method[:start_index] + "<MASK>" + method[end_index:]
    return masked_method

def flatten_and_mask(examples):
    flattened_and_masked_methods = []

    for i in range(len(examples['cleaned_method'])):
        # Flatten method
        flattened_method = flatten(examples['cleaned_method'][i])

        # Format and mask target block
        target_block = examples['target_block'][i][:-2] + ":"
        masked_method = mask_method(flattened_method, target_block)

        flattened_and_masked_methods.append(masked_method)

    examples['flattened_and_masked_method'] = flattened_and_masked_methods
    return examples

dataset = dataset.map(flatten_and_mask, batched=True)

# ------------------------------------------------------------------------------------------------
# 3. Prepare the fine-tuning dataset using the tokenizer we preloaded
# ------------------------------------------------------------------------------------------------
def preprocess_function(examples):
    inputs = examples["flattened_and_masked_method"]
    targets = examples["target_block"]
    # print("Inputs:", inputs[0])
    # print("Targets:", targets[0])
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
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

# ------------------------
# 5. Train the Model
# ------------------------
trainer.train()