from transformers import pipeline

pipe = pipeline("text-generation", model="TheMaster3558/gpt2-ultrachat-finetuned")
result = pipe("What is a good way to learn Python?")
print(result[0]['generated_text'])