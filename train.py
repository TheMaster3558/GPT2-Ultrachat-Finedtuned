from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_from_disk, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from huggingface_hub import login, whoami

# Authenticate with Hugging Face Hub
login()

# Get the username for creating repository names
user_info = whoami()
username = user_info['name']

# Load tokenizer and dataset from Hub
tokenizer = GPT2Tokenizer.from_pretrained(f'{username}/gpt2-ultrachat-tokenizer')
dataset = load_dataset(f'{username}/ultrachat-tokenized-dataset', split='train')
dataset.set_format('torch')

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
epochs = 3
training_steps = epochs * len(train_dataloader)
progress_bar = tqdm(range(training_steps))

optimizer = AdamW(model.parameters(), lr=5e-5)

for _ in range(epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        progress_bar.update(1)

# Push model to Hub
model_repo = f'{username}/gpt2-ultrachat-finetuned'
print(f'Pushing model to {model_repo}...')
model.push_to_hub(model_repo)
print('Model successfully pushed to Hugging Face Hub!')
