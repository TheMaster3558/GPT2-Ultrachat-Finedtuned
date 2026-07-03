from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from huggingface_hub import login, whoami
from dotenv import load_dotenv
import torch

load_dotenv()
login()

user_info = whoami()
username = user_info['name']

tokenizer = GPT2Tokenizer.from_pretrained(f'{username}/gpt2-ultrachat-tokenizer')
dataset = load_dataset(f'{username}/ultrachat-tokenized-dataset', split='train')
dataset.set_format('torch')

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

device = torch.device('cuda')
model = model.to(device)
model = torch.compile(model)

train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
epochs = 3
training_steps = epochs * len(train_dataloader)
progress_bar = tqdm(range(training_steps))

optimizer = AdamW(model.parameters(), lr=5e-5, fused=True)

scaler = torch.amp.GradScaler()

for _ in range(epochs):
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.update(1)


model_repo = f'{username}/gpt2-ultrachat-finetuned'
print(f'Pushing model to {model_repo}...')
model.push_to_hub(model_repo)
print('Model successfully pushed to Hugging Face Hub!')