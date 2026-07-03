from transformers import GPT2LMHeadModel
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from huggingface_hub import login, whoami
from dotenv import load_dotenv
from tokenizer_config import tokenizer
import torch

load_dotenv()
login()

user_info = whoami()
username = user_info['name']

dataset = load_dataset(f'{username}/ultrachat-tokenized-dataset', split='train')
dataset.set_format('torch')

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model.to(device)

train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
epochs = 3
training_steps = epochs * len(train_dataloader)
progress_bar = tqdm(range(training_steps))

optimizer = AdamW(model.parameters(), lr=5e-5)

for _ in range(epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        # Move all tensors in the batch to the chosen device
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        progress_bar.update(1)

model_repo = f'{username}/gpt2-ultrachat-finetuned'
print(f'Pushing model to {model_repo}...')
model.push_to_hub(model_repo)
print('Model successfully pushed to Hugging Face Hub!')
