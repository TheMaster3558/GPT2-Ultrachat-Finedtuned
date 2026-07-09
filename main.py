from transformers import pipeline

pipe = pipeline("text-generation", model="TheMaster3558/gpt2-ultrachat-finetuned")
result = pipe("What is the capital of France?", max_new_tokens=512)
print(result[0]['generated_text'])