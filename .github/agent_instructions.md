# Agent Instructions

## Code Style Guidelines

### String Quotes
- Use single quotes (`'`) for all strings instead of double quotes (`"`)
- This applies to all string literals, print statements, and f-strings
- Example:
  ```python
  # Correct
  message = 'Hello World'
  print(f'{variable} value')
  
  # Incorrect
  message = "Hello World"
  print(f"{variable} value")
  ```

## Project Guidelines

### Workflow Order
- `tokenize_data.py` must be executed first to process and push the tokenizer and dataset to Hugging Face Hub
- `train.py` must be executed second, as it loads the tokenizer and dataset from the Hub

### Hugging Face Hub Repository Naming
- Tokenizer: `{username}/gpt2-ultrachat-tokenizer`
- Dataset: `{username}/ultrachat-tokenized-dataset`
- Model: `{username}/gpt2-ultrachat-finetuned`

### Confirmation Messages
- Always include confirmation messages after pushing to Hugging Face Hub
- Format: `'{resource} successfully pushed to Hugging Face Hub!'`
- Example:
  ```python
  tokenizer.push_to_hub(tokenizer_repo)
  print('Tokenizer successfully pushed to Hugging Face Hub!')
  ```
