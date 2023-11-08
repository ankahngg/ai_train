from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset

# Load the model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


# Tokenize the input text file
def tokenize_function(examples):
    # Tokenize the texts
    tokenized_inputs = tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")

    labels = tokenized_inputs["input_ids"].clone()

    # Replace pad token id's in the labels with -100, which is ignored in loss computation
    labels[labels == tokenizer.pad_token_id] = -100

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Load and prepare dataset
dataset = load_dataset('text', data_files='test.txt')

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask','labels'])

print(tokenized_datasets['train'])


#Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=4,   # batch size per device during training
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    # eval_dataset=tokenized_datasets['validation'] if 'validation' in tokenized_datasets else None,
)

# Train the model
trainer.train()
trainer.save_model("my_fine_tuned_gpt2_model")
tokenizer.save_pretrained("my_fine_tuned_gpt2_model")
