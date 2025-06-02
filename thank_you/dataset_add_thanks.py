from datasets import load_dataset, Dataset
import pandas as pd
import tqdm

# ==== CONFIGURATION ====

DATASET_REPO = 'HuggingFaceH4'
DATASET_NAME = 'ultrachat_200k'
SPLIT = 'train_sft[:10000]'
MY_REPO = 'jdelavande'
MODEL_NAME = 'gemma-2-2b-it'  # change this to reflect your target model

# === CHOOSE YOUR TEMPLATE HERE ===
TEMPLATE = "gemma"  # change to "llama3" or other template key above

# === CHAT TEMPLATE CONFIG ===
chat_templates = {
    "llama3": {
        "bos_token": "<|begin_of_text|>",
        "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "end_token": "<|eot_id|>",
    },
    "mistral": {
        "bos_token": "<s>",
        "user_start": "[INST] ",
        "assistant_start": " ",
        "end_token": " [/INST]",
    },
    "qwen": {
        "bos_token": "",
        "user_start": "<|im_start|>user\n",
        "assistant_start": "<|im_start|>assistant\n",
        "end_token": "<|im_end|>\n",
    },
    "gemma": {
        "bos_token": "",
        "user_start": "<start_of_turn>user\n",
        "assistant_start": "<start_of_turn>model\n",
        "end_token": "<end_of_turn>\n",
    }
}



tpl = chat_templates[TEMPLATE]

# ==== FORMATTER ====

def format_conversation(messages, add_thanks=False):
    formatted = tpl["bos_token"]
    
    for i, msg in enumerate(messages):

        if TEMPLATE == "mistral":
            # Each user message opens [INST] ... [/INST], assistant follows without tag
            if msg["role"] == "user":
                formatted += f'{tpl["user_start"]}{msg["content"].strip()}{tpl["end_token"]}'
            elif msg["role"] == "assistant":
                formatted += f'{tpl["assistant_start"]}{msg["content"].strip()}</s>'

        elif TEMPLATE == "qwen":
            # Qwen-style uses <|im_start|> and <|im_end|> for both user and assistant
            if msg["role"] == "user":
                formatted += f"{tpl['user_start']}{msg['content'].strip()}\n{tpl['end_token']}"
            elif msg["role"] == "assistant":
                formatted += f"{tpl['assistant_start']}{msg['content'].strip()}\n{tpl['end_token']}"

        elif TEMPLATE == "gemma":
            role = "user" if msg["role"] == "user" else "model"
            role_tag = tpl["user_start"] if role == "user" else tpl["assistant_start"]
            formatted += f"{role_tag}{msg['content'].strip()}{tpl['end_token']}"

        else:  # e.g. llama3-style
            if msg["role"] == "user":
                formatted += f'{tpl["user_start"]}{msg["content"].strip()}{tpl["end_token"]}'
            elif msg["role"] == "assistant":
                formatted += f'{tpl["assistant_start"]}{msg["content"].strip()}{tpl["end_token"]}'

    if add_thanks:
        if TEMPLATE == "mistral":
            formatted += f'{tpl["user_start"]}Thank you!{tpl["end_token"]}{tpl["assistant_start"]}'
        elif TEMPLATE == "qwen":
            formatted += f"{tpl['user_start']}Thank you!\n{tpl['end_token']}{tpl['assistant_start']}"
        elif TEMPLATE == "gemma":
            formatted += f"{tpl['user_start']}Thank you!{tpl['end_token']}{tpl['assistant_start']}"
        else:
            formatted += f'{tpl["user_start"]}Thank you!{tpl["end_token"]}{tpl["assistant_start"]}'
    return formatted.strip()

# ==== DATASET CREATION ====

dataset = load_dataset(f"{DATASET_REPO}/{DATASET_NAME}", split=SPLIT)

formatted_data = {
    "conversation_with_thanks": [],
    "conversation_without_thanks": []
}

for sample in tqdm.tqdm(dataset):
    messages = sample["messages"]
    formatted_data["conversation_with_thanks"].append(format_conversation(messages, add_thanks=True))
    formatted_data["conversation_without_thanks"].append(format_conversation(messages, add_thanks=False))

formatted_dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))

# ==== PUSH TO HUB ====

formatted_dataset.push_to_hub(f"{MY_REPO}/{DATASET_NAME}-{MODEL_NAME}-with-thanks")
