# src/utils/helpers.py
import json
from typing import Any

def save_results_to_json(data: Any, filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)