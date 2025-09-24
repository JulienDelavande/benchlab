from datasets import load_dataset, Dataset
import pandas as pd
import tqdm
import argparse

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

def format_conversation(messages: list[dict], template: str, add_thanks: bool = False) -> str:
    """
    Format a conversation into a string based on the specified template.
    Args:
        messages (list[dict]): List of message dictionaries with 'role' and 'content'.
        template (str): Template to use for formatting.
        add_thanks (bool): Whether to append a "Thank you!" message at the end.
    Returns:
        str: Formatted conversation string.
    """
    tpl = chat_templates[template]
    formatted = tpl["bos_token"]

    for msg in messages:

        if template == "mistral":
            # Each user message opens [INST] ... [/INST], assistant follows without tag
            if msg["role"] == "user":
                formatted += f'{tpl["user_start"]}{msg["content"].strip()}{tpl["end_token"]}'
            elif msg["role"] == "assistant":
                formatted += f'{tpl["assistant_start"]}{msg["content"].strip()}</s>'

        elif template == "qwen":
            # Qwen-style uses <|im_start|> and <|im_end|> for both user and assistant
            if msg["role"] == "user":
                formatted += f"{tpl['user_start']}{msg['content'].strip()}\n{tpl['end_token']}"
            elif msg["role"] == "assistant":
                formatted += f"{tpl['assistant_start']}{msg['content'].strip()}\n{tpl['end_token']}"

        elif template == "gemma":
            role = "user" if msg["role"] == "user" else "model"
            role_tag = tpl["user_start"] if role == "user" else tpl["assistant_start"]
            formatted += f"{role_tag}{msg['content'].strip()}{tpl['end_token']}"

        else:  # e.g. llama3-style
            if msg["role"] == "user":
                formatted += f'{tpl["user_start"]}{msg["content"].strip()}{tpl["end_token"]}'
            elif msg["role"] == "assistant":
                formatted += f'{tpl["assistant_start"]}{msg["content"].strip()}{tpl["end_token"]}'

    if add_thanks:
        if template == "mistral":
            formatted += f'{tpl["user_start"]}Thank you!{tpl["end_token"]}{tpl["assistant_start"]}'
        elif template == "qwen":
            formatted += f"{tpl['user_start']}Thank you!\n{tpl['end_token']}{tpl['assistant_start']}"
        elif template == "gemma":
            formatted += f"{tpl['user_start']}Thank you!{tpl['end_token']}{tpl['assistant_start']}"
        else:
            formatted += f'{tpl["user_start"]}Thank you!{tpl["end_token"]}{tpl["assistant_start"]}'
    return formatted.strip()

# ==== DATASET CREATION ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset with and without 'Thank you!' messages.")
    parser.add_argument("--repo", type=str, default="HuggingFaceH4", help="Repository name to push the dataset to.")
    parser.add_argument("--dataset_name", type=str, default="ultrachat_200k", help="Name of the dataset to create.")
    parser.add_argument("--split", type=str, default="train_sft[:10000]", help="Dataset split to use (e.g., 'train', 'test').")
    parser.add_argument("--model_name", type=str, default="gemma-2-2b-it", help="Model name to use for formatting.")
    parser.add_argument("--end_repo", type=str, default="Anonyme162325", help="Repository name to push the dataset to."), 
    parser.add_argument("--template", type=str, default="gemma", choices=list(chat_templates.keys()), help="Template to use for formatting conversations.")
    args = parser.parse_args()

    tpl = chat_templates[args.template]
    dataset = load_dataset(f"{args.repo}/{args.dataset_name}", split=args.split)
    formatted_data = {
        "conversation_with_thanks": [],
        "conversation_without_thanks": []
    }

    for sample in tqdm.tqdm(dataset):
        messages = sample["messages"]
        formatted_data["conversation_with_thanks"].append(format_conversation(messages, add_thanks=True, template=args.template))
        formatted_data["conversation_without_thanks"].append(format_conversation(messages, add_thanks=False, template=args.template))

    formatted_dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    formatted_dataset.push_to_hub(f"{args.end_repo}/{args.dataset_name}-{args.model_name}-with-thanks2")
