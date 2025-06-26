# # With all generation parameters
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --seed 0 --temperature 0 --top_p 1 --use_logits_processor --reasoning_boost 5.0

# # With just temperature
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --temperature 0.1

# # With seed only for reproducibility
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --seed 123

# # Without any generation parameters (original behavior)
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23

# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23

# # With logits processor for reasoning token boost
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --use_logits_processor --reasoning_boost 5.0 --penalty_factor -1000.0

# With logits processor using default settings
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --use_logits_processor

# # With custom boost and penalty factors
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --use_logits_processor --reasoning_boost 10.0 --penalty_factor -500.0

# # Combined with other generation parameters
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --use_logits_processor --temperature 0.7 --seed 42
from datetime import timedelta
import argparse
from pathlib import Path
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available
from token_logits_processor import create_reasoning_token_logits_processor

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate model on math tasks')
    parser.add_argument('--task', type=str, required=True,
                       choices=['gsm8k', 'math', 'math_hard', 'math_500', 'aime24', 'amc23'],
                       help='Task to evaluate')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--use_chat_template', action='store_true', default=False,
                       help='Use chat template')
    parser.add_argument('--override_bs', type=int, default=-1,
                       help='Batch size; -1 for automatic batch size')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for generation')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Temperature for generation (0.0 to 2.0)')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Top-p (nucleus sampling) parameter (0.0 to 1.0)')
    
    # Logits processor arguments
    parser.add_argument('--use_logits_processor', action='store_true', default=False,
                       help='Enable logits processor for reasoning token boost')
    parser.add_argument('--reasoning_boost', type=float, default=5.0,
                       help='Boost factor for reasoning tokens (default: 5.0)')
    parser.add_argument('--penalty_factor', type=float, default=-1000.0,
                       help='Penalty factor for unwanted tokens (default: -1000.0)')
    
    return parser.parse_args()

def main() -> None:
    """Main function to run model evaluation."""
    args = parse_args()
    
    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory="math_verify.tasks",
        use_chat_template=args.use_chat_template,
    )

    model_config_params = {
        "model_name": args.model,
        "dtype": "float32",
        "use_chat_template": args.use_chat_template,
        "max_model_length": 2048
    }
    
    if args.override_bs > 0:
        model_config_params["max_num_seqs"] = args.override_bs

    # Add generation parameters if specified
    generation_params = {}
    if args.seed is not None:
        generation_params["seed"] = args.seed
    if args.temperature is not None:
        generation_params["temperature"] = args.temperature
    if args.top_p is not None:
        generation_params["top_p"] = args.top_p

    # Add logits processor if requested
    if args.use_logits_processor:
        print("Creating logits processor for reasoning token boost...")
        # We need to create the tokenizer first to get token IDs
        from vllm.transformers_utils.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(
            args.model,
            tokenizer_mode="auto",
            trust_remote_code=False,
            revision="main",
        )
        
        logits_processor = create_reasoning_token_logits_processor(
            tokenizer=tokenizer,
            boost_factor=args.reasoning_boost,
            penalty_factor=args.penalty_factor
        )
        generation_params["logits_processors"] = [logits_processor]
        print(f"Logits processor created with boost_factor={args.reasoning_boost}, penalty_factor={args.penalty_factor}")

    if generation_params:
        model_config_params["generation_parameters"] = generation_params

    model_config = VLLMModelConfig(**model_config_params)

    pipeline = Pipeline(
        tasks=f"lighteval|{args.task}|0|1",
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.show_results()
    pipeline.save_and_push_results()

if __name__ == "__main__":
    main()