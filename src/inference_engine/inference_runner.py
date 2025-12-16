# src/inference_engine/inference_runner.py
from typing import List, Dict
from src.api_client.deepseek_client import call_model_batch, call_model_single
from src.prompt_builder.prompt_generator import build_prompt
from src.data_loader.dataset_loader import load_examples

def run_inference(
    questions: List[Dict],
    mode: str,
    submit_mode: str,
    batch_size: int,
    model_config: dict,
    prompt_templates: dict,
    examples_path: str = None
) -> List[Dict]:
    results = []

    # Load few-shot examples from file if path is provided and mode requires examples
    few_shot_examples = []
    if "few" in mode and examples_path:
        few_shot_examples = load_examples(examples_path)
        print(f"[INFO] Loaded {len(few_shot_examples)} examples from {examples_path}")
    elif "few" in mode:
        # Fallback to generating examples from questions if no file path provided
        def generate_few_shot_examples(n=3):
            examples = []
            for i in range(min(n, len(questions))):
                q = questions[i]
                examples.append(f"Q: {q.get('question', '')}\nA: {q.get('answer', '')}")
            return examples
        
        few_shot_examples = generate_few_shot_examples()
        print(f"[INFO] Generated {len(few_shot_examples)} examples from questions")

    if submit_mode == "batch":
        prompts = [
            build_prompt(q["question"], few_shot_examples, mode, prompt_templates)
            for q in questions
        ]
        responses = call_model_batch(prompts, model_config, batch_size=batch_size)

        for idx, response in enumerate(responses):
            results.append({
                "id": questions[idx].get("id"),
                "input_question": questions[idx]["question"],
                "generated_answer": response.strip(),
                "ground_truth": questions[idx].get("answer", "")
            })
    else:
        for q in questions:
            prompt = build_prompt(q["question"], few_shot_examples, mode, prompt_templates)
            response = call_model_single(prompt, model_config)
            results.append({
                "id": q.get("id"),
                "input_question": q["question"],
                "generated_answer": response.strip(),
                "ground_truth": q.get("answer", "")
            })

    return results