#!/usr/bin/env python3
"""
Comprehensive script to run complete pass@k comparison between boosted and unboosted models.
This will run all missing evaluations and create a complete comparison graph.
"""

import subprocess
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def create_output_dir():
    """Create the results_comparison directory if it doesn't exist."""
    output_dir = Path("./results_comparison")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def run_evaluation(k: int, use_logits_processor: bool = False) -> Dict:
    """Run a single evaluation for pass@k with or without logits processor.
    
    Args:
        k: The k value for pass@k evaluation
        use_logits_processor: Whether to use the logits processor (boosted)
        
    Returns:
        Dictionary containing the results
    """
    cmd = [
        "python", "eval.py",
        "--model", "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "--use_chat_template",
        "--task", "amc23",
        "--pass_at_k", "1", str(k),
        "--temperature", "0.7",
        "--top_p", "0.8", 
        "--gpu_memory_utilization", "0.4",
        "--output_dir", "./results_comparison"
    ]
    
    if use_logits_processor:
        cmd.extend(["--use_logits_processor", "--reasoning_boost", "1.2"])
    
    boost_text = "boosted" if use_logits_processor else "unboosted"
    print(f"Running {boost_text} evaluation for pass@{k}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run with verbose output - don't capture stdout/stderr so they show in real-time
        print(f"{'='*70}")
        result = subprocess.run(cmd, timeout=3600)  # 1 hour timeout, show output in real-time
        print(f"{'='*70}")
        
        if result.returncode != 0:
            print(f"✗ Error: Evaluation failed with return code {result.returncode}")
            return None
        
        print(f"✓ Completed {boost_text} evaluation for pass@{k}")
        return {"success": True}
    
    except subprocess.TimeoutExpired:
        print(f"✗ Evaluation timed out for k={k}, boosted={use_logits_processor}")
        return None
    except Exception as e:
        print(f"✗ Error running evaluation: {e}")
        return None

def extract_pass_at_k_score(results_dir: Path, task_name: str) -> float:
    """Extract pass@k score from results directory.
    
    Args:
        results_dir: Path to results directory
        task_name: Name of the task (e.g., "amc23_pass@1:8")
        
    Returns:
        The pass@k score as a float
    """
    # Look for JSON files in the results directory structure
    model_dir = results_dir / "results" / "Qwen" / "Qwen2.5-Math-1.5B-Instruct"
    if not model_dir.exists():
        print(f"Model directory {model_dir} does not exist")
        return None
        
    json_files = list(model_dir.glob("*.json"))
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Sort by modification time, newest first
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Look for the task results
            if 'results' in data:
                for task_key, task_result in data['results'].items():
                    if task_name in task_key and isinstance(task_result, dict):
                        # Extract the score
                        score_key = f"pass@1:{task_name.split(':')[1]}_samples"
                        if score_key in task_result:
                            score = float(task_result[score_key])
                            print(f"Found score {score:.3f} for {task_name} in {json_file.name}")
                            return score
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    print(f"Could not find score for {task_name}")
    return None

def extract_all_comparison_results(results_dir: Path):
    """Extract all pass@k results from the comparison results directory."""
    model_dir = results_dir / "results" / "Qwen" / "Qwen2.5-Math-1.5B-Instruct"
    
    if not model_dir.exists():
        print(f"Model directory {model_dir} does not exist")
        return {}, {}
    
    boosted_results = defaultdict(list)
    unboosted_results = defaultdict(list)
    
    json_files = list(model_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Check if it's a pass@k evaluation for amc23
            if 'results' not in data:
                continue
                
            for task_key, task_result in data['results'].items():
                if 'amc23_pass@1:' in task_key and isinstance(task_result, dict):
                    # Extract k value
                    k_str = task_key.split('amc23_pass@1:')[1].split('|')[0]
                    try:
                        k = int(k_str)
                    except ValueError:
                        continue
                    
                    # Extract score
                    score_key = f"pass@1:{k}_samples"
                    if score_key in task_result:
                        score = float(task_result[score_key])
                        
                        # Check if boosted or unboosted
                        generation_params = data.get('config_general', {}).get('generation_parameters', {})
                        logits_processors = generation_params.get('logits_processors')
                        
                        if logits_processors is None or logits_processors == []:
                            unboosted_results[k].append(score)
                        else:
                            boosted_results[k].append(score)
                        
                        boost_text = "boosted" if (logits_processors is not None and logits_processors != []) else "unboosted"
                        print(f"Found: k={k}, score={score:.3f}, {boost_text}")
                        
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    return boosted_results, unboosted_results

def get_best_scores(results_dict):
    """Get the best (highest) score for each k value."""
    best_scores = {}
    for k, scores in results_dict.items():
        if scores:
            best_scores[k] = max(scores)
    return best_scores

def create_comparison_graph(k_values: List[int], unboosted_scores: List[float], boosted_scores: List[float]):
    """Create a comparison graph of pass@k performance."""
    plt.figure(figsize=(12, 8))
    
    # Plot both lines with enhanced styling
    plt.plot(k_values, unboosted_scores, 'o-', label='PPO', linewidth=3, markersize=10, color='#4A4A4A', alpha=0.8)
    plt.plot(k_values, boosted_scores, 'o-', label='PPO w/ Entropy Advantage', linewidth=3, markersize=10, color='#8A2BE2', alpha=0.8)
    
    plt.xlabel('k', fontsize=16, fontweight='bold')
    plt.ylabel('Pass@K Performance', fontsize=16, fontweight='bold')
    plt.title('Pass@K Performance', fontsize=18, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    
    # Set x-axis to show all k values
    plt.xticks(k_values, fontsize=14)
    plt.yticks(fontsize=14)
    
    # Set y-axis limits
    all_values = unboosted_scores + boosted_scores
    min_score = min(all_values)
    max_score = max(all_values)
    margin = (max_score - min_score) * 0.1
    plt.ylim(max(0, min_score - margin), min(1, max_score + margin))
    
    # Add value labels on points
    for i, (k, unboosted, boosted) in enumerate(zip(k_values, unboosted_scores, boosted_scores)):
        plt.annotate(f'{unboosted:.3f}', (k, unboosted), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10, alpha=0.7)
        plt.annotate(f'{boosted:.3f}', (k, boosted), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the graph
    plt.savefig('pass_at_k_comparison_complete.png', dpi=300, bbox_inches='tight')
    plt.savefig('pass_at_k_comparison_complete.pdf', bbox_inches='tight')
    
    print(f"\nGraph saved as pass_at_k_comparison_complete.png and pass_at_k_comparison_complete.pdf")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("COMPLETE PASS@K COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'k':>4} {'PPO':>12} {'PPO w/ Entropy':>16} {'Improvement':>14} {'Status':>10}")
    print(f"{'-'*80}")
    
    for i, k in enumerate(k_values):
        unboosted = unboosted_scores[i]
        boosted = boosted_scores[i]
        improvement = ((boosted - unboosted) / unboosted * 100) if unboosted > 0 else 0
        status = "✓ Better" if improvement > 0 else "✗ Worse" if improvement < 0 else "= Same"
        print(f"{k:>4} {unboosted:>12.3f} {boosted:>16.3f} {improvement:>13.1f}% {status:>10}")
    
    # Save results to JSON
    results = {
        "k_values": k_values,
        "unboosted_scores": unboosted_scores,
        "boosted_scores": boosted_scores,
        "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "task": "amc23",
        "reasoning_boost": 1.2,
        "temperature": 0.7,
        "top_p": 0.8,
        "source": "complete_comparison"
    }
    
    with open('pass_at_k_results_complete.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to pass_at_k_results_complete.json")

def main():
    """Main function to run complete pass@k comparison."""
    target_k_values = [1, 2, 4, 8, 16, 32, 64]
    
    print("="*80)
    print("COMPLETE PASS@K COMPARISON: BOOSTED vs UNBOOSTED")
    print("="*80)
    
    # Create output directory
    results_dir = create_output_dir()
    print(f"Results will be saved to: {results_dir}")
    
    # Run all evaluations
    for k in target_k_values:
        print(f"\n{'='*60}")
        print(f"EVALUATING PASS@{k}")
        print(f"{'='*60}")
        
        # Run unboosted evaluation
        unboosted_result = run_evaluation(k, use_logits_processor=False)
        if unboosted_result is None:
            print(f"⚠️ Skipping k={k} due to unboosted evaluation failure")
            continue
            
        time.sleep(3)  # Small delay between evaluations
        
        # Run boosted evaluation  
        boosted_result = run_evaluation(k, use_logits_processor=True)
        if boosted_result is None:
            print(f"⚠️ Skipping k={k} due to boosted evaluation failure")
            continue
            
        time.sleep(5)  # Delay between different k values
        print(f"✓ Completed all evaluations for k={k}")
    
    # Extract all results and create comparison
    print(f"\n{'='*60}")
    print("EXTRACTING RESULTS AND CREATING COMPARISON")
    print(f"{'='*60}")
    
    time.sleep(5)  # Wait for files to be fully written
    
    boosted_results, unboosted_results = extract_all_comparison_results(results_dir)
    
    # Get best scores for each k
    best_boosted = get_best_scores(boosted_results)
    best_unboosted = get_best_scores(unboosted_results)
    
    # Get common k values
    common_k_values = sorted(set(best_boosted.keys()) & set(best_unboosted.keys()))
    
    if not common_k_values:
        print("❌ No common k values found between boosted and unboosted results!")
        return
    
    boosted_values = [best_boosted[k] for k in common_k_values]
    unboosted_values = [best_unboosted[k] for k in common_k_values]
    
    # Create comparison graph
    create_comparison_graph(common_k_values, unboosted_values, boosted_values)
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 