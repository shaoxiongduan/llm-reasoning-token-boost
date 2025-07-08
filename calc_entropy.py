# Entropy-based evaluation for language models
# Based on "Reasoning with Exploration: An Entropy Perspective" and related work
# Captures token-level log probabilities and calculates entropy during generation

from datetime import timedelta
import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import torch
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available

# Add Math-Verify-main/src to Python path so math_verify can be imported
sys.path.insert(0, str(Path("Math-Verify-main/src").absolute()))

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

class EntropyAnalyzer:
    """Analyzer for calculating token-level entropy and related metrics."""
    
    def __init__(self):
        self.entropy_data = []
        self.token_data = []
        
    def calculate_entropy(self, logprobs: List[Dict[str, float]]) -> List[float]:
        """Calculate entropy for each token position from log probabilities.
        
        Args:
            logprobs: List of dicts containing token -> logprob mappings for each position
            
        Returns:
            List of entropy values for each token position
        """
        entropies = []
        for token_logprobs in logprobs:
            if not token_logprobs:
                entropies.append(0.0)
                continue
                
            # Convert logprobs to probabilities
            logprob_values = list(token_logprobs.values())
            if not logprob_values:
                entropies.append(0.0)
                continue
                
            # Convert from log probabilities to probabilities
            logprob_tensor = torch.tensor(logprob_values, dtype=torch.float32)
            probs = torch.exp(logprob_tensor)
            
            # Normalize probabilities (they should already be normalized, but just in case)
            probs = probs / probs.sum()
            
            # Calculate entropy: H = -sum(p * log(p))
            # Use natural logarithm for entropy calculation
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            entropies.append(entropy.item())
            
        return entropies
    
    def analyze_token_patterns(self, tokens: List[str], entropies: List[float]) -> Dict[str, Any]:
        """Analyze patterns in token entropy for exploratory reasoning.
        
        Args:
            tokens: List of generated tokens
            entropies: List of entropy values for each token
            
        Returns:
            Dictionary containing analysis results
        """
        # Identify pivotal tokens (high entropy connectors)
        pivotal_keywords = [
            'therefore', 'however', 'because', 'since', 'thus', 'hence',
            'first', 'second', 'then', 'next', 'finally', 'but', 'so',
            'moreover', 'furthermore', 'additionally', 'consequently'
        ]
        
        # Identify reflective tokens (self-verification patterns)
        reflective_keywords = [
            'wait', 'actually', 'let me', 'check', 'verify', 'correct',
            'mistake', 'wrong', 'recalculate', 'review', 'double-check'
        ]
        
        analysis = {
            'avg_entropy': np.mean(entropies) if entropies else 0.0,
            'max_entropy': max(entropies) if entropies else 0.0,
            'min_entropy': min(entropies) if entropies else 0.0,
            'entropy_std': np.std(entropies) if entropies else 0.0,
            'high_entropy_tokens': [],
            'pivotal_tokens': [],
            'reflective_tokens': [],
            'token_entropy_pairs': list(zip(tokens, entropies))
        }
        
        # Find high entropy tokens (top 10% or above mean + std)
        if entropies:
            threshold = np.mean(entropies) + np.std(entropies)
            for i, (token, entropy) in enumerate(zip(tokens, entropies)):
                if entropy > threshold:
                    analysis['high_entropy_tokens'].append({
                        'token': token,
                        'entropy': entropy,
                        'position': i
                    })
                    
                # Check for pivotal tokens
                token_lower = token.lower().strip()
                # Remove leading/trailing punctuation and whitespace for matching
                cleaned_token = token_lower.strip(' .,!?;:()[]{}"\'-')
                if cleaned_token in pivotal_keywords:
                    analysis['pivotal_tokens'].append({
                        'token': token,
                        'entropy': entropy,
                        'position': i
                    })
                    
                # Check for reflective tokens (use substring matching for multi-word phrases)
                if any(keyword in token.lower() for keyword in reflective_keywords):
                    analysis['reflective_tokens'].append({
                        'token': token,
                        'entropy': entropy,
                        'position': i
                    })
        
        return analysis
    
    def save_entropy_data(self, output_file: str):
        """Save collected entropy data to file."""
        with open(output_file, 'w') as f:
            json.dump({
                'entropy_data': self.entropy_data,
                'token_data': self.token_data
            }, f, indent=2)
    
    def process_pipeline_results(self, pipeline_results: Dict[str, Any], tokenizer=None) -> None:
        """Process results from lighteval pipeline to extract entropy data."""
        if not hasattr(pipeline_results, 'details') or not pipeline_results.details:
            print("Warning: No detailed results found in pipeline output")
            return
            
        # Extract entropy data from pipeline results
        # This will depend on how lighteval stores the logprobs
        for task_name, task_results in pipeline_results.details.items():
            for result in task_results:
                if hasattr(result, 'logprobs') and result.logprobs:
                    # Extract tokens and logprobs
                    tokens = []
                    logprobs_data = []
                    
                    # Process logprobs from lighteval result
                    if isinstance(result.logprobs, list):
                        for token_logprobs in result.logprobs:
                            if token_logprobs:
                                # Find the token with highest probability
                                if tokenizer:
                                    max_token_id = max(token_logprobs.keys(), key=lambda k: token_logprobs[k])
                                    try:
                                        token_id_int = int(max_token_id) if isinstance(max_token_id, str) else max_token_id
                                        token_text = tokenizer.decode([token_id_int])
                                        tokens.append(token_text)
                                    except (ValueError, TypeError):
                                        tokens.append(f"token_{max_token_id}")
                                
                                # Convert to string -> logprob dict
                                token_logprobs_str = {}
                                for token_id, logprob in token_logprobs.items():
                                    if tokenizer:
                                        try:
                                            token_id_int = int(token_id) if isinstance(token_id, str) else token_id
                                            token_text = tokenizer.decode([token_id_int])
                                            token_logprobs_str[token_text] = logprob
                                        except (ValueError, TypeError):
                                            token_logprobs_str[str(token_id)] = logprob
                                    else:
                                        token_logprobs_str[str(token_id)] = logprob
                                
                                logprobs_data.append(token_logprobs_str)
                    
                    # Calculate entropy
                    entropies = self.calculate_entropy(logprobs_data)
                    
                    # Analyze patterns
                    analysis = self.analyze_token_patterns(tokens, entropies)
                    
                    # Store data
                    entry_data = {
                        'prompt': result.prompt if hasattr(result, 'prompt') else '',
                        'generated_text': result.generated_text if hasattr(result, 'generated_text') else '',
                        'tokens': tokens,
                        'entropies': entropies,
                        'analysis': analysis,
                        'logprobs': logprobs_data
                    }
                    
                    self.entropy_data.append(entry_data)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate model on math tasks with entropy analysis')
    parser.add_argument('--task', type=str, required=True,
                       choices=['gsm8k', 'math', 'math_hard', 'math_500', 'aime24', 'amc23'],
                       help='Task to evaluate')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--use_chat_template', action='store_true', default=False,
                       help='Use chat template')
    parser.add_argument('--output_dir', type=str, default='./entropy_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for generation (0.0 to 2.0)')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p (nucleus sampling) parameter (0.0 to 1.0)')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.5,
                       help='GPU memory utilization fraction')
    parser.add_argument('--min_token_occurrences', type=int, default=5,
                       help='Minimum number of occurrences for a token to be shown in frequent high-entropy tokens analysis')
    
    return parser.parse_args()

def main():
    """Main function for entropy-based evaluation."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize entropy analyzer
    entropy_analyzer = EntropyAnalyzer()
    
    # Use original task name like eval.py
    task_name = args.task
    
    # Set up evaluation tracker
    evaluation_tracker = EvaluationTracker(
        output_dir=str(output_dir),
        save_details=True,
        push_to_hub=False,
    )

    # Set up pipeline parameters (same as eval.py)
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory="math_verify",  # Same as eval.py
        use_chat_template=args.use_chat_template,
    )

    # Set up model configuration (same pattern as eval.py)
    model_config_params = {
        "model_name": args.model,
        "dtype": "bfloat16",  # Match eval.py
        "use_chat_template": args.use_chat_template,
        "max_model_length": 2048,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }

    # Add generation parameters including logprobs (following eval.py pattern)
    generation_params = {}
    if args.seed is not None:
        generation_params["seed"] = args.seed
    if args.temperature is not None:
        generation_params["temperature"] = args.temperature
    if args.top_p is not None:
        generation_params["top_p"] = args.top_p
    
    # Set generation length to 2048 tokens (same as eval.py)
    generation_params["max_new_tokens"] = args.max_new_tokens
    
    # Add logprobs for entropy calculation
    generation_params["logprobs"] = 20  # Enable logprobs for entropy calculation

    if generation_params:
        model_config_params["generation_parameters"] = generation_params

    model_config = VLLMModelConfig(**model_config_params)

    # Create and run pipeline
    print(f"Running entropy analysis on {task_name} dataset with logprobs enabled...")
    pipeline = Pipeline(
        tasks=f"lighteval|{task_name}|0|1",
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    # Run evaluation
    pipeline.evaluate()
    
    # Get results and extract entropy data
    results = pipeline.get_results()
    
    # Get tokenizer for entropy analysis
    tokenizer = None
    if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'tokenizer'):
        tokenizer = pipeline.model.tokenizer
    elif hasattr(pipeline, 'model') and hasattr(pipeline.model, 'get_tokenizer'):
        tokenizer = pipeline.model.get_tokenizer()
    
    print("Extracting entropy data from pipeline results...")
    print(f"Pipeline results type: {type(results)}")
    print(f"Pipeline results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
    
    # Debug: Print the structure of results
    if hasattr(results, '__dict__'):
        print(f"Results attributes: {list(results.__dict__.keys())}")
    
    # Try to access evaluation tracker details
    print(f"Evaluation tracker type: {type(evaluation_tracker)}")
    if hasattr(evaluation_tracker, 'details'):
        print(f"Evaluation tracker has details: {evaluation_tracker.details is not None}")
        if evaluation_tracker.details:
            print(f"Details keys: {list(evaluation_tracker.details.keys())}")
    
    # Process results to extract entropy data
    entropy_analyzer.process_pipeline_results(results, tokenizer)
    
    # If no entropy data was extracted, try alternative approach
    if not entropy_analyzer.entropy_data:
        print("No entropy data found in pipeline results. Trying alternative extraction...")
        
        # Try to access detailed results from different sources
        detailed_results = None
        
        # Check evaluation tracker
        if hasattr(evaluation_tracker, 'details') and evaluation_tracker.details:
            detailed_results = evaluation_tracker.details
            print(f"Found details in evaluation_tracker: {len(detailed_results)} tasks")
            
        # Check if results has details  
        elif hasattr(results, 'details') and results.details:
            detailed_results = results.details
            print(f"Found details in results: {len(detailed_results)} tasks")
            
        # Check if results has a details attribute or method
        elif isinstance(results, dict) and 'details' in results:
            detailed_results = results['details']
            print(f"Found details in results dict: {len(detailed_results)} tasks")
            
        # Try to access the model's generation outputs directly
        if detailed_results:
            # Dataset-level collections
            all_dataset_entropies = []
            all_dataset_tokens = []
            all_dataset_logprobs = []
            processed_problems = 0
            total_problems = 0
            correct_problems = 0
            
            print("Collecting entropy data across entire dataset (correct answers only)...")
            
            for task_name, task_results in detailed_results.items():
                print(f"Processing task: {task_name} with {len(task_results)} results")
                total_problems = len(task_results)
                
                for i, result in enumerate(task_results):
                    # Since results are dictionaries, process them
                    if isinstance(result, dict):
                        # Check if this result is correct (only process entropy for correct answers)
                        is_correct = False
                        if 'metrics' in result and isinstance(result['metrics'], dict):
                            extractive_match = result['metrics'].get('extractive_match', 0)
                            is_correct = extractive_match == 1
                        
                        if is_correct:
                            correct_problems += 1
                        else:
                            continue  # Skip incorrect answers
                        # Extract entropy data from pred_logits
                        if 'pred_logits' in result and result['pred_logits'] and 'predictions' in result and result['predictions']:
                            pred_logits = result['pred_logits']
                            predictions = result['predictions']
                            
                            if isinstance(pred_logits, list) and len(pred_logits) > 0 and predictions:
                                logits_data = pred_logits[0]  # First prediction's logits
                                prediction_text = predictions[0]  # First prediction
                                
                                if isinstance(logits_data, list) and len(logits_data) > 0:
                                    # Check if we have full logprobs (new format) or just selected token logprobs (old format)
                                    if isinstance(logits_data[0], dict):
                                        # New format: [{token_id: logprob, ...}, ...] - full distribution per token
                                        
                                        # Calculate entropy from full distributions
                                        for token_pos, token_logprobs in enumerate(logits_data):
                                            if token_logprobs:
                                                # Convert logprobs to probabilities and calculate entropy
                                                logprob_values = list(token_logprobs.values())
                                                if logprob_values:
                                                    # Convert from log probabilities to probabilities
                                                    logprob_tensor = torch.tensor(logprob_values, dtype=torch.float32)
                                                    probs = torch.exp(logprob_tensor)
                                                    
                                                    # Normalize probabilities (they should already be normalized)
                                                    probs = probs / probs.sum()
                                                    
                                                    # Calculate entropy: H = -sum(p * log(p))
                                                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                                                    all_dataset_entropies.append(entropy.item())
                                                    
                                                    # Get the token string for the highest probability token
                                                    max_token_id = max(token_logprobs.keys(), key=lambda k: token_logprobs[k])
                                                    if tokenizer:
                                                        # Handle both string and integer token IDs
                                                        try:
                                                            token_id_int = int(max_token_id) if isinstance(max_token_id, str) else max_token_id
                                                            token_text = tokenizer.decode([token_id_int])
                                                            all_dataset_tokens.append(token_text)
                                                        except (ValueError, TypeError):
                                                            all_dataset_tokens.append(f"token_{max_token_id}")
                                                    else:
                                                        all_dataset_tokens.append(f"token_{max_token_id}")
                                                    
                                                    # Store full logprobs for this token
                                                    all_dataset_logprobs.append(token_logprobs)
                                                else:
                                                    all_dataset_entropies.append(0.0)
                                                    all_dataset_tokens.append("[UNK]")
                                                    all_dataset_logprobs.append({})
                                            else:
                                                all_dataset_entropies.append(0.0)
                                                all_dataset_tokens.append("[UNK]")
                                                all_dataset_logprobs.append({})
                                        
                                        processed_problems += 1
                                        
                                    else:
                                        # Old format: [logprob1, logprob2, ...] - only selected token logprobs
                                        print(f"! Found old format logprobs for result {i} - skipping")
                                        continue
                                else:
                                    print(f"✗ Invalid logits data for result {i}")
                            else:
                                print(f"✗ No predictions or pred_logits for result {i}")
                        else:
                            print(f"✗ No pred_logits or predictions found for result {i}")
                    
                    # Progress update every 10 problems
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(task_results)} problems...")
                
                break  # Only process first task for now
            
            # Analyze dataset-level patterns
            print(f"\nAnalyzing dataset-level entropy patterns...")
            print(f"Total problems: {total_problems}")
            print(f"Correct problems: {correct_problems}")
            print(f"Accuracy: {correct_problems/total_problems*100:.1f}%")
            print(f"Problems processed for entropy: {processed_problems}")
            print(f"Total tokens collected: {len(all_dataset_entropies)}")
            
            if all_dataset_entropies:
                # Perform dataset-level analysis
                dataset_analysis = entropy_analyzer.analyze_token_patterns(all_dataset_tokens, all_dataset_entropies)
                
                # Create single dataset-level entropy data entry
                dataset_entropy_data = {
                    'dataset': args.task,
                    'model': args.model,
                    'total_problems': total_problems,
                    'correct_problems': correct_problems,
                    'accuracy': correct_problems/total_problems if total_problems > 0 else 0,
                    'num_problems_processed': processed_problems,
                    'total_tokens': len(all_dataset_entropies),
                    'tokens': all_dataset_tokens,
                    'entropies': all_dataset_entropies,
                    'logprobs_full': all_dataset_logprobs,
                    'analysis': dataset_analysis
                }
                
                entropy_analyzer.entropy_data.append(dataset_entropy_data)
                
                print(f"✓ Dataset-level entropy analysis complete")
                print(f"  Total entropy values: {len(all_dataset_entropies)}")
                print(f"  Entropy range: {min(all_dataset_entropies):.4f} to {max(all_dataset_entropies):.4f}")
                print(f"  Average entropy: {np.mean(all_dataset_entropies):.4f}")
                print(f"  Entropy std: {np.std(all_dataset_entropies):.4f}")
                print(f"  High entropy tokens: {len(dataset_analysis['high_entropy_tokens'])}")
                print(f"  Pivotal tokens found: {len(dataset_analysis['pivotal_tokens'])}")
                print(f"  Reflective tokens found: {len(dataset_analysis['reflective_tokens'])}")
            else:
                print("✗ No entropy data collected from dataset")
        else:
            print("Warning: No detailed results available for entropy analysis")
            print("This may be due to lighteval pipeline configuration")
    
    # Final dataset analysis and reporting
    print("\n" + "="*60)
    print("DATASET-LEVEL ENTROPY ANALYSIS")
    print("="*60)
    
    if entropy_analyzer.entropy_data and len(entropy_analyzer.entropy_data) > 0:
        dataset_data = entropy_analyzer.entropy_data[0]  # Single dataset entry
        analysis = dataset_data['analysis']
        all_entropies = dataset_data['entropies']
        
        # Print comprehensive dataset statistics
        print(f"Dataset: {dataset_data['dataset']}")
        print(f"Model: {dataset_data['model']}")
        print(f"Total problems: {dataset_data['total_problems']}")
        print(f"Correct problems: {dataset_data['correct_problems']}")
        print(f"Accuracy: {dataset_data['accuracy']*100:.1f}%")
        print(f"Problems processed for entropy: {dataset_data['num_problems_processed']}")
        print(f"Total tokens: {dataset_data['total_tokens']}")
        print(f"")
        print(f"ENTROPY STATISTICS:")
        print(f"  Average entropy: {np.mean(all_entropies):.4f}")
        print(f"  Entropy std: {np.std(all_entropies):.4f}")
        print(f"  Min entropy: {min(all_entropies):.4f}")
        print(f"  Max entropy: {max(all_entropies):.4f}")
        print(f"  Median entropy: {np.median(all_entropies):.4f}")
        print(f"")
        print(f"TOKEN ANALYSIS:")
        print(f"  High entropy tokens: {len(analysis['high_entropy_tokens'])}")
        print(f"  Pivotal tokens: {len(analysis['pivotal_tokens'])}")
        print(f"  Reflective tokens: {len(analysis['reflective_tokens'])}")
        
        # Count frequent high-entropy tokens for summary
        frequent_count = 0
        if analysis['high_entropy_tokens']:
            token_occurrences = {}
            for token_info in analysis['high_entropy_tokens']:
                token_text = token_info['token']
                if token_text not in token_occurrences:
                    token_occurrences[token_text] = []
                token_occurrences[token_text].append(token_info['entropy'])
            
            for token_text, entropies in token_occurrences.items():
                if len(entropies) >= args.min_token_occurrences:
                    frequent_count += 1
        
        print(f"  Frequent high-entropy tokens (≥{args.min_token_occurrences} occurrences): {frequent_count}")
        
        # Calculate entropy distribution percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"")
        print(f"ENTROPY DISTRIBUTION:")
        for p in percentiles:
            value = np.percentile(all_entropies, p)
            print(f"  {p}th percentile: {value:.4f}")
        
        # Count tokens by entropy ranges
        very_low = sum(1 for e in all_entropies if e < 0.1)
        low = sum(1 for e in all_entropies if 0.1 <= e < 0.5)
        medium = sum(1 for e in all_entropies if 0.5 <= e < 1.0)
        high = sum(1 for e in all_entropies if 1.0 <= e < 1.5)
        very_high = sum(1 for e in all_entropies if e >= 1.5)
        
        print(f"")
        print(f"ENTROPY RANGES:")
        print(f"  Very low (< 0.1): {very_low} tokens ({100*very_low/len(all_entropies):.1f}%)")
        print(f"  Low (0.1-0.5): {low} tokens ({100*low/len(all_entropies):.1f}%)")
        print(f"  Medium (0.5-1.0): {medium} tokens ({100*medium/len(all_entropies):.1f}%)")
        print(f"  High (1.0-1.5): {high} tokens ({100*high/len(all_entropies):.1f}%)")
        print(f"  Very high (≥1.5): {very_high} tokens ({100*very_high/len(all_entropies):.1f}%)")
        
        # Show top high-entropy tokens
        high_entropy_tokens = analysis['high_entropy_tokens']
        if high_entropy_tokens:
            print(f"")
            print(f"TOP HIGH-ENTROPY TOKENS:")
            
            # Count occurrences of each token text
            token_occurrences = {}
            for token_info in high_entropy_tokens:
                token_text = token_info['token']
                if token_text not in token_occurrences:
                    token_occurrences[token_text] = []
                token_occurrences[token_text].append(token_info['entropy'])
            
            # Create list of unique tokens with their stats, sorted by average entropy
            unique_tokens = []
            for token_text, entropies in token_occurrences.items():
                unique_tokens.append({
                    'token': token_text,
                    'occurrences': len(entropies),
                    'avg_entropy': sum(entropies) / len(entropies),
                    'max_entropy': max(entropies)
                })
            
            # Sort by average entropy descending
            unique_tokens.sort(key=lambda x: x['avg_entropy'], reverse=True)
            
            # Show top 50 with occurrence counts and average entropy
            for count, token_info in enumerate(unique_tokens[:50]):
                print(f"  {count+1}. '{token_info['token']}' - avg entropy: {token_info['avg_entropy']:.4f}, max entropy: {token_info['max_entropy']:.4f}, occurrences: {token_info['occurrences']}")
            
            # Show frequent high-entropy tokens (appearing more than k times)
            print(f"")
            print(f"FREQUENT HIGH-ENTROPY TOKENS (≥{args.min_token_occurrences} occurrences):")
            
            # Filter tokens by minimum occurrences and sort by average entropy
            frequent_tokens = []
            for token_text, entropies in token_occurrences.items():
                if len(entropies) >= args.min_token_occurrences:
                    frequent_tokens.append({
                        'token': token_text,
                        'occurrences': len(entropies),
                        'avg_entropy': np.mean(entropies),
                        'max_entropy': max(entropies),
                        'min_entropy': min(entropies),
                        'std_entropy': np.std(entropies)
                    })
            
            # Sort by average entropy descending
            frequent_tokens.sort(key=lambda x: x['avg_entropy'], reverse=True)
            
            if frequent_tokens:
                print(f"  Found {len(frequent_tokens)} tokens with ≥{args.min_token_occurrences} occurrences:")
                for i, token_info in enumerate(frequent_tokens[:30]):  # Show top 30
                    print(f"  {i+1}. '{token_info['token']}' - avg entropy: {token_info['avg_entropy']:.4f}, "
                          f"max: {token_info['max_entropy']:.4f}, occurrences: {token_info['occurrences']}, "
                          f"std: {token_info['std_entropy']:.4f}")
            else:
                print(f"  No tokens found with ≥{args.min_token_occurrences} occurrences.")
        
        # Show pivotal tokens
        pivotal_tokens = analysis['pivotal_tokens']
        if pivotal_tokens:
            print(f"")
            print(f"PIVOTAL TOKENS (reasoning connectors):")
            # Group by token text and show average entropy
            pivotal_by_text = {}
            for token_info in pivotal_tokens:
                text = token_info['token']
                if text not in pivotal_by_text:
                    pivotal_by_text[text] = []
                pivotal_by_text[text].append(token_info['entropy'])
            
            for text, entropies in pivotal_by_text.items():
                avg_entropy = np.mean(entropies)
                count = len(entropies)
                print(f"  '{text}': {count} occurrences, avg entropy: {avg_entropy:.4f}")
        
        # Save dataset-level results
        entropy_file = output_dir / f"{args.task}_dataset_entropy.json"
        entropy_analyzer.save_entropy_data(str(entropy_file))
        print(f"\nDataset entropy data saved to: {entropy_file}")
        
        # Save dataset summary
        summary_file = output_dir / f"{args.task}_dataset_summary.json"
        
        # Calculate frequent tokens data for summary
        frequent_tokens_for_summary = []
        if high_entropy_tokens:
            token_occurrences = {}
            for token_info in high_entropy_tokens:
                token_text = token_info['token']
                if token_text not in token_occurrences:
                    token_occurrences[token_text] = []
                token_occurrences[token_text].append(token_info['entropy'])
            
            for token_text, entropies in token_occurrences.items():
                if len(entropies) >= args.min_token_occurrences:
                    frequent_tokens_for_summary.append({
                        'token': token_text,
                        'occurrences': len(entropies),
                        'avg_entropy': float(np.mean(entropies)),
                        'max_entropy': float(max(entropies)),
                        'min_entropy': float(min(entropies)),
                        'std_entropy': float(np.std(entropies))
                    })
            
            # Sort by average entropy descending
            frequent_tokens_for_summary.sort(key=lambda x: x['avg_entropy'], reverse=True)
        
        summary_data = {
            'model': args.model,
            'dataset': args.task,
            'total_problems': dataset_data['total_problems'],
            'correct_problems': dataset_data['correct_problems'],
            'accuracy': dataset_data['accuracy'],
            'num_problems_processed': dataset_data['num_problems_processed'],
            'total_tokens': dataset_data['total_tokens'],
            'min_token_occurrences_threshold': args.min_token_occurrences,
            'entropy_stats': {
                'mean': float(np.mean(all_entropies)),
                'std': float(np.std(all_entropies)),
                'min': float(min(all_entropies)),
                'max': float(max(all_entropies)),
                'median': float(np.median(all_entropies)),
                'percentiles': {f'p{p}': float(np.percentile(all_entropies, p)) for p in percentiles}
            },
            'token_counts': {
                'high_entropy': len(analysis['high_entropy_tokens']),
                'pivotal': len(analysis['pivotal_tokens']),
                'reflective': len(analysis['reflective_tokens']),
                'frequent_high_entropy': len(frequent_tokens_for_summary)
            },
            'entropy_ranges': {
                'very_low': very_low,
                'low': low,
                'medium': medium,
                'high': high,
                'very_high': very_high
            },
            'frequent_high_entropy_tokens': frequent_tokens_for_summary[:30],  # Top 30 for summary
            'generation_params': generation_params
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Dataset summary saved to: {summary_file}")
        
        print(f"\nDataset-level entropy analysis complete!")
    else:
        print("No entropy data was extracted. This may be due to:")
        print("1. Lighteval pipeline not capturing logprobs properly")
        print("2. vLLM model not returning logprobs in expected format")
        print("3. Pipeline configuration issues")
        print("\nTry checking the lighteval and vLLM documentation for logprobs support.")
    
    # Show standard evaluation results
    print("\n" + "="*50)
    print("Standard Evaluation Results:")
    pipeline.show_results()

if __name__ == "__main__":
    main()
