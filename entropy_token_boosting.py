#!/usr/bin/env python3
"""
Entropy-based token boosting utilities.
Extract top k highest entropy tokens from merged entropy results and use them for token boosting.
"""

import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from token_logits_processor import TokenLogitsProcessor, find_token_ids


def load_merged_entropy_results(file_path: str) -> Dict[str, Any]:
    """
    Load merged entropy results from a JSON file.
    
    Args:
        file_path: Path to the merged entropy results JSON file
        
    Returns:
        Dictionary containing merged entropy results
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Merged entropy results file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")


def extract_top_k_entropy_tokens(merged_results: Dict[str, Any], k: int = 20, min_occurrences: int = 5) -> List[Dict[str, Any]]:
    """
    Extract the top k highest entropy tokens from merged entropy results.
    
    Args:
        merged_results: Dictionary containing merged entropy results
        k: Number of top tokens to extract
        min_occurrences: Minimum number of occurrences required for a token to be considered
        
    Returns:
        List of token dictionaries sorted by average entropy (descending)
        Each dict contains: token, token_id, avg_entropy, occurrences, etc.
    """
    if 'frequent_high_entropy_tokens' not in merged_results:
        print("Warning: No 'frequent_high_entropy_tokens' found in merged results")
        return []
    
    tokens = merged_results['frequent_high_entropy_tokens']
    
    # Filter by minimum occurrences
    filtered_tokens = [
        token for token in tokens 
        if token.get('occurrences', 0) >= min_occurrences
    ]
    
    print(f"Found {len(tokens)} total tokens, {len(filtered_tokens)} with >= {min_occurrences} occurrences")
    
    # Sort by average entropy (descending) and take top k
    sorted_tokens = sorted(filtered_tokens, key=lambda x: x.get('avg_entropy', 0), reverse=True)
    top_k_tokens = sorted_tokens[:k]
    
    print(f"Selected top {len(top_k_tokens)} highest entropy tokens:")
    for i, token in enumerate(top_k_tokens[:10]):  # Show first 10
        print(f"  {i+1:2d}. '{token['token']:15s}' | "
              f"Entropy: {token['avg_entropy']:.4f} | "
              f"Occurrences: {token['occurrences']:3d} | "
              f"Token ID: {token.get('token_id', 'N/A')}")
    
    if len(top_k_tokens) > 10:
        print(f"  ... and {len(top_k_tokens) - 10} more tokens")
    
    return top_k_tokens


def create_entropy_based_logits_processor(
    tokenizer, 
    merged_entropy_file: str,
    top_k: int = 20,
    min_occurrences: int = 5,
    boost_factor: float = 1.2,
    penalty_factor: float = 1.0,
    include_unwanted_tokens: bool = True
) -> TokenLogitsProcessor:
    """
    Create a logits processor that boosts tokens based on entropy analysis.
    
    Args:
        tokenizer: The tokenizer to use for finding token IDs
        merged_entropy_file: Path to merged entropy results JSON file
        top_k: Number of top entropy tokens to boost
        min_occurrences: Minimum occurrences for a token to be considered
        boost_factor: Boost factor for high entropy tokens
        penalty_factor: Penalty factor for unwanted tokens (like <|endoftext|>)
        include_unwanted_tokens: Whether to include penalty for unwanted tokens
        
    Returns:
        TokenLogitsProcessor: Configured logits processor
    """
    print(f"Creating entropy-based logits processor from: {merged_entropy_file}")
    print(f"Parameters: top_k={top_k}, min_occurrences={min_occurrences}, boost_factor={boost_factor}")
    
    # Load merged entropy results
    try:
        merged_results = load_merged_entropy_results(merged_entropy_file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading entropy results: {e}")
        print("Falling back to empty token boosting")
        return TokenLogitsProcessor(token_id_boosts={})
    
    # Extract top k entropy tokens
    top_entropy_tokens = extract_top_k_entropy_tokens(merged_results, k=top_k, min_occurrences=min_occurrences)
    
    if not top_entropy_tokens:
        print("Warning: No entropy tokens found, creating processor without boosts")
        return TokenLogitsProcessor(token_id_boosts={})
    
    # Extract token strings from entropy results
    entropy_token_strings = [token['token'] for token in top_entropy_tokens]
    
    print(f"\nUsing {len(entropy_token_strings)} entropy-based tokens for boosting:")
    for token in entropy_token_strings:
        print(f"  '{token}'")
    
    # Find token IDs for entropy tokens
    entropy_mappings = find_token_ids(tokenizer, entropy_token_strings)
    
    token_id_boosts = {}
    
    # Add boosts for entropy tokens
    boost_count = 0
    for token_str, token_ids in entropy_mappings.items():
        for token_id in token_ids:
            token_id_boosts[token_id] = boost_factor
            boost_count += 1
    
    print(f"Added boosts for {boost_count} token IDs")
    
    # Add penalties for unwanted tokens (optional)
    if include_unwanted_tokens:
        unwanted_strings = ["<|endoftext|>"]
        unwanted_mappings = find_token_ids(tokenizer, unwanted_strings)
        
        penalty_count = 0
        for token_ids in unwanted_mappings.values():
            for token_id in token_ids:
                token_id_boosts[token_id] = penalty_factor
                penalty_count += 1
        
        print(f"Added penalties for {penalty_count} unwanted token IDs")
    
    print(f"\nFinal entropy-based token_id_boosts: {len(token_id_boosts)} total mappings")
    
    return TokenLogitsProcessor(token_id_boosts=token_id_boosts)


def analyze_entropy_tokens(merged_entropy_file: str, top_k: int = 50) -> None:
    """
    Analyze and display information about entropy tokens from merged results.
    
    Args:
        merged_entropy_file: Path to merged entropy results JSON file
        top_k: Number of top tokens to analyze
    """
    print(f"Analyzing entropy tokens from: {merged_entropy_file}")
    print("="*80)
    
    try:
        merged_results = load_merged_entropy_results(merged_entropy_file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading entropy results: {e}")
        return
    
    # Show summary information
    if 'summary_stats' in merged_results:
        stats = merged_results['summary_stats']
        print("SUMMARY STATISTICS:")
        print(f"  Entropy mean: {stats['avg_entropy_mean']:.4f} ± {stats['avg_entropy_std']:.4f}")
        print(f"  Entropy range: [{stats['avg_entropy_min']:.4f}, {stats['avg_entropy_max']:.4f}]")
        print(f"  Total occurrences: {stats['total_occurrences']}")
        print()
    
    # Extract and display top k tokens
    top_tokens = extract_top_k_entropy_tokens(merged_results, k=top_k, min_occurrences=1)
    
    if not top_tokens:
        print("No entropy tokens found!")
        return
    
    print(f"TOP {len(top_tokens)} ENTROPY TOKENS:")
    print("-" * 80)
    print(f"{'Rank':>4} {'Token':>20} {'Entropy':>10} {'Occurrences':>12} {'Token ID':>10}")
    print("-" * 80)
    
    for i, token in enumerate(top_tokens):
        print(f"{i+1:4d} {repr(token['token']):>20} "
              f"{token['avg_entropy']:10.4f} "
              f"{token['occurrences']:12d} "
              f"{token.get('token_id', 'N/A'):>10}")
    
    # Show distribution by entropy ranges
    print(f"\nDISTRIBUTION BY ENTROPY RANGES:")
    print("-" * 40)
    
    ranges = [
        (0.0, 0.5, "Very Low"),
        (0.5, 1.0, "Low"), 
        (1.0, 1.5, "Medium"),
        (1.5, 2.0, "High"),
        (2.0, float('inf'), "Very High")
    ]
    
    for min_entropy, max_entropy, label in ranges:
        count = sum(1 for token in top_tokens 
                   if min_entropy <= token['avg_entropy'] < max_entropy)
        print(f"  {label:>10}: {count:3d} tokens")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Analyze entropy tokens and create entropy-based token boosting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python entropy_token_boosting.py merged_results.json --analyze --top-k 30
  python entropy_token_boosting.py merged_results.json --show-tokens --top-k 20 --min-occurrences 10
        """
    )
    
    parser.add_argument(
        'merged_file',
        help='Path to merged entropy results JSON file'
    )
    
    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Analyze and display entropy token information'
    )
    
    parser.add_argument(
        '--show-tokens', '-s',
        action='store_true', 
        help='Show top entropy tokens that would be used for boosting'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=20,
        help='Number of top entropy tokens to use (default: 20)'
    )
    
    parser.add_argument(
        '--min-occurrences', '-m',
        type=int,
        default=5,
        help='Minimum occurrences for a token to be considered (default: 5)'
    )
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_entropy_tokens(args.merged_file, top_k=args.top_k)
    
    if args.show_tokens:
        print(f"\nTokens that would be used for boosting (top_k={args.top_k}, min_occurrences={args.min_occurrences}):")
        print("="*80)
        
        try:
            merged_results = load_merged_entropy_results(args.merged_file)
            top_tokens = extract_top_k_entropy_tokens(merged_results, k=args.top_k, min_occurrences=args.min_occurrences)
            
            if top_tokens:
                for i, token in enumerate(top_tokens):
                    print(f"{i+1:2d}. '{token['token']:15s}' | "
                          f"Entropy: {token['avg_entropy']:.4f} | "
                          f"Occurrences: {token['occurrences']:3d}")
            else:
                print("No tokens found matching criteria")
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: {e}")
    
    if not (args.analyze or args.show_tokens):
        print("Use --analyze or --show-tokens to see information about entropy tokens")
        print("Use --help for more options")


if __name__ == "__main__":
    main() 