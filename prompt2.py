from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained("my_fine_tuned_gpt2_model")
model = GPT2LMHeadModel.from_pretrained("my_fine_tuned_gpt2_model")

# Encode some text to be used as context
input_text = "hung is a"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text until the output length (which includes the context length) reaches 100
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode the generated ids to a readable text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
