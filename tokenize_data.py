from transformers import GPT2Tokenizer
from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens': [
    '<|system|>',
    '<|user|>',
    '<|assistant|>'
]})

dataset = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft')#.shuffle(seed=42).select(range(20000))

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
        tokenizer.convert_tokens_to_ids("<|system|>"),
        tokenizer.convert_tokens_to_ids("<|user|>"),
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

if __name__ == '__main__':
    import os

    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names, load_from_cache_file=False)
    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names, num_proc=max(1, os.cpu_count() - 1), load_from_cache_file=False)
    dataset = dataset.map(mask_user_prompts, load_from_cache_file=False)

    tokenizer.save_pretrained('./saved_tokenizer')
    dataset.save_to_disk('./saved_dataset')
