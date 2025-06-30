# # With all generation parameters
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --seed 0 --temperature 0 --top_p 1 --use_logits_processor --reasoning_boost 1.15

# # With just temperature
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --temperature 0.1

# # With seed only for reproducibility
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --seed 123

# # Without any generation parameters (original behavior)
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23

# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23

# # With logits processor for reasoning token boost
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --use_logits_processor --reasoning_boost 1.2 --penalty_factor -1000.0

# With logits processor using default settings
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --use_logits_processor

# # With custom boost and penalty factors
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --use_logits_processor --reasoning_boost 1.1 --penalty_factor -500.0

# # Combined with other generation parameters
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --use_logits_processor --temperature 0.7 --seed 42



# USE THESE COMMANDS FOR EVAL

# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --temperature 0.7 --top_p 0.8
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --temperature 0.7 --top_p 0.8 --use_logits_processor --reasoning_boost 1.1

# NEW: MULTI-GENERATION EVALUATION COMMANDS
# Now supports three different multi-generation evaluation metrics:
# 
# 1. avg@n: Computes the average correctness across n generations per problem
#    - For a problem with 3/5 correct generations: avg@5 gives score = 3/5 = 0.6
#    - Equivalent to running pass@1 evaluation n times and averaging the results
#
# 2. pass@k: Checks if ANY of k generations is correct (from n total generations)
#    - For a problem with 3/5 correct generations: pass@3 gives score = 1.0 (since ≥3 are correct)
#    - Uses the standard pass@k formula from code generation literature
#
# 3. maj@n: Uses majority voting across n generations
#    - Takes the most common answer among n generations and checks if it's correct
#    - For a problem where 3/5 generations give answer "A" and it's correct: maj@5 gives score = 1.0
#
# Examples for avg@n (average correctness across n generations):
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --avg_at_n 4
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --avg_at_n 8 --temperature 0.7
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --avg_at_n 16 --use_logits_processor --reasoning_boost 1.1

# Examples for pass@k (pass if any of k generations is correct):
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --pass_at_k 1 4  # pass@1 with 4 generations
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --pass_at_k 3 8 --temperature 0.7  # pass@3 with 8 generations
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --pass_at_k 5 16 --use_logits_processor --reasoning_boost 1.1  # pass@5 with 16 generations

# Examples for maj@n (majority voting across n generations):
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --maj_at_n 4
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --maj_at_n 8 --temperature 0.7
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --maj_at_n 16 --use_logits_processor --reasoning_boost 1.1

# MEMORY MANAGEMENT COMMANDS (for systems with limited VRAM):
# For systems with ~12GB available VRAM (like yours with other processes using half the memory):
# python eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --use_chat_template --task amc23 --avg_at_n 4 --temperature 0.7 --top_p 0.8 --gpu_memory_utilization 0.5 --max_num_seqs 32 --max_num_batched_tokens 1024

from datetime import timedelta
import argparse
import sys
from pathlib import Path
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available
from token_logits_processor import create_reasoning_token_logits_processor

# Add Math-Verify-main/src to Python path so math_verify can be imported
sys.path.insert(0, str(Path("Math-Verify-main/src").absolute()))

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
    
    # Multi-generation evaluation metrics (mutually exclusive)
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument('--avg_at_n', type=int, default=None,
                             choices=[4, 8, 16],
                             help='Enable avg@n evaluation (average correctness across n generations)')
    metric_group.add_argument('--pass_at_k', type=int, nargs=2, default=None,
                             metavar=('K', 'N'),
                             help='Enable pass@k evaluation: K N (pass if any of K generations correct from N total)')
    metric_group.add_argument('--maj_at_n', type=int, default=None,
                             choices=[4, 8, 16],
                             help='Enable maj@n evaluation (majority voting across n generations)')
    
    # Logits processor arguments
    parser.add_argument('--use_logits_processor', action='store_true', default=False,
                       help='Enable logits processor for reasoning token boost')
    parser.add_argument('--reasoning_boost', type=float, default=5.0,
                       help='Boost factor for reasoning tokens (default: 5.0)')
    parser.add_argument('--penalty_factor', type=float, default=1.0,
                       help='Penalty factor for unwanted tokens (default: 1.0)')
    
    # Memory management arguments
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.5,
                       help='GPU memory utilization fraction (0.0 to 1.0, default: 0.5 for memory-constrained systems)')
    parser.add_argument('--max_num_seqs', type=int, default=32,
                       help='Maximum number of sequences per batch (default: 32 for lower memory usage)')
    parser.add_argument('--max_num_batched_tokens', type=int, default=1024,
                       help='Maximum number of batched tokens (default: 1024 for lower memory usage)')
    
    args = parser.parse_args()
    
    # Validate pass@k arguments
    if args.pass_at_k:
        k, n = args.pass_at_k
        if k > n:
            parser.error("In pass@k evaluation, K must be <= N")
        if k <= 0 or n <= 0:
            parser.error("In pass@k evaluation, both K and N must be positive")
        if n not in [4, 8, 16]:
            parser.error("In pass@k evaluation, N must be one of [4, 8, 16]")
    
    return args

def main() -> None:
    """Main function to run model evaluation."""
    args = parse_args()
    
    # Determine task name and evaluation type based on metric settings
    if args.avg_at_n:
        task_name = f"{args.task}_avg@{args.avg_at_n}"
        print(f"Running avg@{args.avg_at_n} evaluation (average correctness across {args.avg_at_n} generations per problem)")
    elif args.pass_at_k:
        k, n = args.pass_at_k
        task_name = f"{args.task}_pass@{k}:{n}"
        print(f"Running pass@{k} evaluation (pass if any of {k} generations correct from {n} total generations per problem)")
    elif args.maj_at_n:
        task_name = f"{args.task}_maj@{args.maj_at_n}"
        print(f"Running maj@{args.maj_at_n} evaluation (majority voting across {args.maj_at_n} generations per problem)")
    else:
        task_name = args.task
        print("Running single-generation evaluation")
    
    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory="math_verify",
        use_chat_template=args.use_chat_template,
    )

    model_config_params = {
        "model_name": args.model,
        "dtype": "float32",
        "use_chat_template": args.use_chat_template,
        "max_model_length": 2048,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        # "max_num_seqs": args.max_num_seqs,
        # "max_num_batched_tokens": args.max_num_batched_tokens,
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
    
    # Set generation length to 2048 tokens
    generation_params["max_new_tokens"] = 2048

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
        tasks=f"lighteval|{task_name}|0|1",
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.show_results()
    pipeline.save_and_push_results()

if __name__ == "__main__":
    main()