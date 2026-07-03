from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


tokenizer = GPT2Tokenizer.from_pretrained('./saved_tokenizer')
dataset = load_from_disk('./saved_dataset')
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

model.save_pretrained('./saved_model')
