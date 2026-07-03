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

### Tokenizer Configuration
- Tokenizer setup is stored in `tokenizer_config.py`
- Both `tokenize_data.py` and `train.py` import the tokenizer from this file: `from tokenizer_config import tokenizer`
- Do not create or push the tokenizer to the Hub; it's a shared configuration

### Workflow Order
- `tokenize_data.py` must be executed first to process and push the dataset to Hugging Face Hub
- `train.py` must be executed second, as it loads the dataset from the Hub

### Hugging Face Hub Repository Naming
- Dataset: `{username}/ultrachat-tokenized-dataset`
- Model: `{username}/gpt2-ultrachat-finetuned`

