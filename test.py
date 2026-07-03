from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import login, whoami
from dotenv import load_dotenv
import torch
import math
from tqdm.auto import tqdm

load_dotenv()
login()

user_info = whoami()
username = user_info['name']

model = GPT2LMHeadModel.from_pretrained(f'{username}/gpt2-ultrachat-finetuned')
tokenizer = GPT2Tokenizer.from_pretrained(f'{username}/gpt2-ultrachat-finetuned')

dataset = load_dataset(f'{username}/ultrachat-tokenized-dataset', split='test')
dataset.set_format('torch')

test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()


# Calculate metrics
total_loss = 0.0
total_tokens = 0
all_logits = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Evaluating'):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        total_loss += loss.item() * batch['input_ids'].size(0)
        total_tokens += batch['input_ids'].size(0)

# Calculate average loss and perplexity
avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)

print('=' * 50)
print('TEST METRICS')
print('=' * 50)
print(f'Average Loss: {avg_loss:.4f}')
print(f'Perplexity: {perplexity:.4f}')
print(f'Total Tokens Evaluated: {total_tokens}')
print('=' * 50)

