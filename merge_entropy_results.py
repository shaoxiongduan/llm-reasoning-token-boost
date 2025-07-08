#!/usr/bin/env python3
"""
Merge multiple entropy analysis JSON files and calculate combined distribution statistics.
Only keeps tokens that appear in ALL input files (intersection).
"""

import json
import argparse
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict


def load_entropy_file(file_path: str) -> Dict[str, Any]:
    """Load a single entropy analysis JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
        return None


def merge_token_data(token_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple token entries with the same token string.
    Combines occurrences and recalculates statistics.
    """
    if not token_entries:
        return {}
    
    # Use the first entry as base
    merged = token_entries[0].copy()
    
    if len(token_entries) == 1:
        return merged
    
    # Collect all individual entropy values (weighted by occurrences)
    all_entropy_values = []
    total_occurrences = 0
    all_positions = []
    max_entropy = float('-inf')
    min_entropy = float('inf')
    
    for entry in token_entries:
        occurrences = entry['occurrences']
        avg_entropy = entry['avg_entropy']
        
        # Approximate individual entropy values (this is an approximation since we don't have raw data)
        # We'll use the average entropy repeated for each occurrence
        all_entropy_values.extend([avg_entropy] * occurrences)
        
        total_occurrences += occurrences
        all_positions.extend(entry.get('positions', []))
        
        max_entropy = max(max_entropy, entry['max_entropy'])
        min_entropy = min(min_entropy, entry['min_entropy'])
    
    # Calculate combined statistics
    all_entropy_values = np.array(all_entropy_values)
    
    merged.update({
        'occurrences': total_occurrences,
        'avg_entropy': np.mean(all_entropy_values),
        'max_entropy': max_entropy,
        'min_entropy': min_entropy,
        'std_entropy': np.std(all_entropy_values),
        'median_entropy': np.median(all_entropy_values),
        'positions': sorted(all_positions)
    })
    
    return merged


def merge_entropy_files(file_paths: List[str], output_path: str = None) -> Dict[str, Any]:
    """
    Merge multiple entropy analysis JSON files.
    
    Args:
        file_paths: List of paths to JSON files
        output_path: Optional path to save merged results
    
    Returns:
        Dictionary with merged results
    """
    if not file_paths:
        print("Error: No file paths provided")
        return {}
    
    # Load all files
    all_data = []
    datasets = []
    models = []
    
    for file_path in file_paths:
        print(f"Loading {file_path}...")
        data = load_entropy_file(file_path)
        if data is None:
            continue
        all_data.append(data)
        datasets.append(data.get('dataset', 'unknown'))
        models.append(data.get('model', 'unknown'))
    
    if not all_data:
        print("Error: No valid files loaded")
        return {}
    
    # Group tokens by token string and token_id
    token_groups = defaultdict(list)
    
    for data in all_data:
        for token_entry in data.get('frequent_high_entropy_tokens', []):
            # Use both token string and token_id as key to ensure exact matching
            key = (token_entry['token'], token_entry['token_id'])
            token_groups[key].append(token_entry)
    
    # Only keep tokens that appear in ALL input files
    num_files = len(all_data)
    common_tokens = {key: entries for key, entries in token_groups.items() 
                    if len(entries) == num_files}
    
    print(f"Found {len(token_groups)} unique tokens across all files")
    print(f"Found {len(common_tokens)} tokens that appear in all {num_files} files")
    
    # Merge tokens (only those that appear in all files)
    merged_tokens = []
    for (token_str, token_id), token_entries in common_tokens.items():
        merged_token = merge_token_data(token_entries)
        merged_tokens.append(merged_token)
    
    # Sort by average entropy (descending) then by occurrences (descending)
    merged_tokens.sort(key=lambda x: (-x['avg_entropy'], -x['occurrences']))
    
    # Create merged result
    merged_result = {
        'datasets': list(set(datasets)),
        'models': list(set(models)),
        'source_files': file_paths,
        'total_frequent_tokens': len(merged_tokens),
        'frequent_high_entropy_tokens': merged_tokens
    }
    
    # Add summary statistics
    if merged_tokens:
        all_avg_entropies = [token['avg_entropy'] for token in merged_tokens]
        all_occurrences = [token['occurrences'] for token in merged_tokens]
        
        merged_result['summary_stats'] = {
            'avg_entropy_mean': np.mean(all_avg_entropies),
            'avg_entropy_std': np.std(all_avg_entropies),
            'avg_entropy_median': np.median(all_avg_entropies),
            'avg_entropy_min': np.min(all_avg_entropies),
            'avg_entropy_max': np.max(all_avg_entropies),
            'total_occurrences': sum(all_occurrences),
            'occurrences_mean': np.mean(all_occurrences),
            'occurrences_std': np.std(all_occurrences),
            'occurrences_median': np.median(all_occurrences)
        }
    
    # Save to file if output path provided
    if output_path:
        print(f"Saving merged results to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(merged_result, f, indent=2)
        print(f"Merged results saved to {output_path}")
    
    return merged_result


def print_summary(merged_result: Dict[str, Any]):
    """Print a summary of the merged results."""
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    
    print(f"Datasets: {', '.join(merged_result['datasets'])}")
    print(f"Models: {', '.join(merged_result['models'])}")
    print(f"Source files: {len(merged_result['source_files'])}")
    print(f"Common tokens (appearing in all files): {merged_result['total_frequent_tokens']}")
    
    if 'summary_stats' in merged_result:
        stats = merged_result['summary_stats']
        print(f"\nSummary Statistics (Common Tokens Only):")
        print(f"  Average Entropy: {stats['avg_entropy_mean']:.4f} ± {stats['avg_entropy_std']:.4f}")
        print(f"  Entropy Range: [{stats['avg_entropy_min']:.4f}, {stats['avg_entropy_max']:.4f}]")
        print(f"  Total Occurrences: {stats['total_occurrences']}")
        print(f"  Avg Occurrences: {stats['occurrences_mean']:.2f} ± {stats['occurrences_std']:.2f}")
    
    print(f"\nTop 10 Common Tokens by Average Entropy:")
    print("-" * 60)
    for i, token in enumerate(merged_result['frequent_high_entropy_tokens'][:10]):
        print(f"{i+1:2d}. '{token['token']:15s}' | "
              f"Entropy: {token['avg_entropy']:.4f} | "
              f"Occurrences: {token['occurrences']:3d}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple entropy analysis JSON files (only keeps tokens that appear in ALL files)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_entropy_results.py file1.json file2.json file3.json
  python merge_entropy_results.py *.json --output merged_results.json
  python merge_entropy_results.py dataset1.json dataset2.json --output combined.json --summary
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='Paths to entropy analysis JSON files to merge'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path for merged results (optional)'
    )
    
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Print detailed summary of merged results'
    )
    
    args = parser.parse_args()
    
    # Merge files
    merged_result = merge_entropy_files(args.files, args.output)
    
    if not merged_result:
        print("Error: No results to merge")
        return 1
    
    # Print summary if requested or if no output file specified
    if args.summary or not args.output:
        print_summary(merged_result)
    
    return 0


if __name__ == "__main__":
    exit(main()) 