from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load trained model
model_path = "./gpt2-finetuned"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Prompt
prompt = "Artificial Intelligence"

inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    inputs,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:")
print(generated_text)