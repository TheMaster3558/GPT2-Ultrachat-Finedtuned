import psutil
from datasets import load_dataset, DatasetDict
from huggingface_hub import login, whoami
from dotenv import load_dotenv
from transformers import GPT2Tokenizer

load_dotenv()
login()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens': [
    '<|system|>',
    '<|user|>',
    '<|assistant|>'
]})

raw_datasets = load_dataset('HuggingFaceH4/ultrachat_200k')
train_dataset = raw_datasets['train_sft'].shuffle().select(range(50000))
test_dataset = raw_datasets['test_sft'].shuffle().select(range(5000))

def format_conversation(example):
    text = ''
    for message in example['messages']:
        text += f'<|{message['role']}|>\n{message['content']}\n'
    return {'messages': text}

def tokenize(examples):
    return tokenizer(examples['messages'], truncation=True, padding='max_length', max_length=512)

def mask_user_prompts(example):
    input_ids = example['input_ids']
    labels = [-100] * len(input_ids)

    assistant_token_id = tokenizer.convert_tokens_to_ids('<|assistant|>')
    other_role_token_ids = {
        tokenizer.convert_tokens_to_ids('<|system|>'),
        tokenizer.convert_tokens_to_ids('<|user|>'),
    }

    in_assistant_span = False
    for i, token_id in enumerate(input_ids):
        if token_id == assistant_token_id:
            in_assistant_span = True
            continue
        if token_id in other_role_token_ids:
            in_assistant_span = False
        if in_assistant_span:
            labels[i] = input_ids[i]

    return {'labels': labels}

def process_dataset(dataset):
    physical = psutil.cpu_count(logical=False)
    num_proc = max(1, physical // 2)

    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names, load_from_cache_file=False)
    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names, num_proc=num_proc, load_from_cache_file=False)
    dataset = dataset.map(mask_user_prompts, load_from_cache_file=False)
    return dataset

if __name__ == '__main__':
    user_info = whoami()
    username = user_info['name']

    train_dataset = process_dataset(train_dataset)
    test_dataset = process_dataset(test_dataset)

    processed_datasets = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # Push tokenizer to Hub
    tokenizer_repo = f'{username}/gpt2-ultrachat-finetuned'
    print(f'Pushing tokenizer to {tokenizer_repo}...')
    tokenizer.push_to_hub(tokenizer_repo)
    print('Tokenizer successfully pushed to Hugging Face Hub!')

    # Push dataset to Hub
    dataset_repo = f'{username}/ultrachat-tokenized-dataset'
    print(f'Pushing dataset to {dataset_repo}...')
    processed_datasets.push_to_hub(dataset_repo)
    print('Dataset successfully pushed to Hugging Face Hub!')
