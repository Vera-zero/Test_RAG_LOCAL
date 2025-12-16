# src/api_client/deepseek_client.py
import requests
from typing import List

def call_model_single(prompt: str, config: dict) -> str:
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"]
    }
    resp = requests.post(config["api_base_url"] + "/chat/completions", json=payload, headers=headers)
    return resp.json()["choices"][0]["message"]["content"]

def call_model_batch(prompts: List[str], config: dict, batch_size: int = 4) -> List[str]:
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_payloads = [{
            "model": config["model_name"],
            "messages": [{"role": "user", "content": p}],
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"]
        } for p in batch_prompts]

        # Simulate parallel processing (简化版)
        batch_responses = [call_model_single(p, config) for p in batch_prompts]
        responses.extend(batch_responses)
    return responses