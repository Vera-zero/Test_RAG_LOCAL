# main.py
import os
import yaml
import argparse
from typing import List, Dict
from src.data_loader.dataset_loader import load_questions
from src.prompt_builder.prompt_generator import build_prompt
from src.inference_engine.inference_runner import run_inference
from src.utils.helpers import save_results_to_json

def load_yaml_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run QA inference on datasets.")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "zero_shot_cot", "few_shot_cot"],
                        default="zero_shot", help="Inference mode")
    parser.add_argument("--dataset", default="all", help="Dataset to run ('all' for all)")
    parser.add_argument("--submit_mode", choices=["single", "batch"], default="single")
    parser.add_argument("--batch_size", type=int, default=4)
    
    args = parser.parse_args()

    # Load configs
    model_config = load_yaml_config("config/model_config.yaml")
    dataset_config = load_yaml_config("config/dataset_config.yaml")
    prompt_templates = load_yaml_config("config/prompt_template.yaml")

    datasets_to_run = dataset_config["datasets"]
    if args.dataset != "all":
        datasets_to_run = [d for d in datasets_to_run if d["name"] == args.dataset]

    for dataset_info in datasets_to_run:
        print(f"[INFO] Running {args.mode} on {dataset_info['name']}...")
        
        questions = load_questions(dataset_info["question_path"])
        questions = questions[:2] #测试用例
        
        # Get examples path if available
        examples_path = dataset_info.get("examples_path")
        
        results = run_inference(
            questions=questions,
            mode=args.mode,
            submit_mode=args.submit_mode,
            batch_size=args.batch_size,
            model_config=model_config,
            prompt_templates=prompt_templates,
            examples_path=examples_path
        )

        output_path = f"./outputs/{dataset_info['name']}_{args.mode}.json"
        save_results_to_json(results, output_path)
        print(f"[DONE] Results saved to {output_path}")

if __name__ == "__main__":
    main()