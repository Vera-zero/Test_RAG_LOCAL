# src/data_loader/dataset_loader.py
import json
from typing import List, Dict

def load_questions(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_examples(file_path: str) -> List[str]:
    """Load few-shot examples from a JSON file.
    
    Expected format:
    [
        {"question": "Example question 1", "answer": "Answer 1"},
        {"question": "Example question 2", "answer": "Answer 2"}
    ]
    
    Returns:
        List of formatted example strings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            examples_data = json.load(f)
        
        examples = []
        for item in examples_data:
            examples.append(f"Q: {item['question']}\nA: {item['answer']}")
        
        return examples
    except FileNotFoundError:
        print(f"[WARNING] Examples file not found at {file_path}, using empty examples list")
        return []
    except Exception as e:
        print(f"[WARNING] Error loading examples from {file_path}: {e}, using empty examples list")
        return []