# src/api_client/deepseek_client.py
import requests
from typing import List
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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
    """
    批量处理提示词
    
    Args:
        prompts: 需要处理的提示词列表
        config: API配置字典
        batch_size: 每批次处理的提示词数量
    
    Returns:
        响应结果列表，与输入提示词一一对应
    """
    responses = []
    
    # 分批处理所有提示词
    for i in range(0, len(prompts), batch_size):
        # 获取当前批次的提示词
        batch_prompts = prompts[i:i+batch_size]
        
        # 使用线程池实现真正的并行处理
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # 使用executor.map保持结果顺序与输入顺序一致
            batch_responses = list(executor.map(
                lambda prompt: call_model_single(prompt, config),
                batch_prompts
            ))
            
        # 将当前批次的响应添加到总响应列表中
        responses.extend(batch_responses)
    
    return responses