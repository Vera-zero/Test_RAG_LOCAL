# src/prompt_builder/prompt_generator.py
from typing import List, Dict

def build_prompt(question: str, examples: List[str], mode: str, templates: dict) -> str:
    if mode == "zero_shot":
        return templates["zero_shot_template"].format(question=question)
    elif mode == "few_shot":
        example_text = "\n".join(examples)
        return templates["few_shot_template"].format(examples=example_text, question=question)
    elif mode == "zero_shot_cot":
        return templates["zero_shot_cot_template"].format(question=question)
    elif mode == "few_shot_cot":
        example_text = "\n".join(examples)
        return templates["few_shot_cot_template"].format(examples=example_text, question=question)
    else:
        raise ValueError(f"Unsupported mode: {mode}")