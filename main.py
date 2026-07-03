from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TheMaster3558/gpt2-ultrachat-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    min_new_tokens=20,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=False))  # note: NOT skipping special tokens this time