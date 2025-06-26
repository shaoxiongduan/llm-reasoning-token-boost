#!/usr/bin/env python3
"""
Simplified but robust analysis script for lighteval evaluation results.
Focuses on essential analytics while avoiding JSON serialization issues.
"""

import pandas as pd
import json
import re
import numpy as np
from collections import Counter
from datetime import datetime


def analyze_text_length(text):
    """Analyze text length in characters, words, and estimated tokens."""
    if pd.isna(text) or text is None or text == "":
        return {"chars": 0, "words": 0, "estimated_tokens": 0}
    
    text_str = str(text)
    char_count = len(text_str)
    word_count = len(text_str.split())
    estimated_tokens = char_count / 4  # Rough estimation
    
    return {
        "chars": char_count,
        "words": word_count,
        "estimated_tokens": int(estimated_tokens)
    }


def extract_text_from_array(arr):
    """Extract text from numpy array or list."""
    if isinstance(arr, np.ndarray):
        if arr.size > 0:
            return str(arr[0]) if len(arr) > 0 else ""
        return ""
    elif isinstance(arr, list):
        return str(arr[0]) if len(arr) > 0 else ""
    else:
        return str(arr) if arr else ""


def get_word_frequency(texts):
    """Get word frequency analysis from a list of texts."""
    all_words = []
    for text in texts:
        if text and str(text).strip():
            words = re.findall(r'\b[a-zA-Z]+\b', str(text).lower())
            all_words.extend(words)
    
    return Counter(all_words)


def compare_word_frequencies(freq1, freq2, run1_name, run2_name, top_n=20):
    """Compare word frequencies between two runs and find the most different tokens."""
    # Get all unique words from both runs
    all_words = set(freq1.keys()) | set(freq2.keys())
    
    # Calculate frequency differences
    frequency_diffs = []
    
    for word in all_words:
        count1 = freq1.get(word, 0)
        count2 = freq2.get(word, 0)
        
        # Calculate absolute difference and relative difference
        abs_diff = count1 - count2
        
        # Avoid division by zero and calculate relative difference
        if count2 == 0 and count1 > 0:
            rel_diff = float('inf')  # Infinite relative difference
        elif count1 == 0 and count2 > 0:
            rel_diff = float('-inf')  # Negative infinite relative difference
        elif count2 == 0 and count1 == 0:
            rel_diff = 0
        else:
            rel_diff = (count1 - count2) / max(count1, count2)
        
        frequency_diffs.append({
            'word': word,
            'count_run1': count1,
            'count_run2': count2,
            'absolute_diff': abs_diff,
            'relative_diff': rel_diff
        })
    
    # Sort by absolute difference (descending)
    frequency_diffs_abs = sorted(frequency_diffs, key=lambda x: abs(x['absolute_diff']), reverse=True)
    
    # Get words that appear much more in run1 vs run2
    more_in_run1 = sorted([x for x in frequency_diffs if x['absolute_diff'] > 0], 
                         key=lambda x: x['absolute_diff'], reverse=True)
    
    # Get words that appear much more in run2 vs run1
    more_in_run2 = sorted([x for x in frequency_diffs if x['absolute_diff'] < 0], 
                         key=lambda x: abs(x['absolute_diff']), reverse=True)
    
    return {
        'most_different_overall': frequency_diffs_abs[:top_n],
        'more_frequent_in_run1': more_in_run1[:top_n],
        'more_frequent_in_run2': more_in_run2[:top_n],
        'run1_name': run1_name,
        'run2_name': run2_name
    }


def analyze_evaluation_data(df, run_name):
    """Analyze evaluation data and return analytics."""
    print(f"\nAnalyzing {run_name}...")
    
    # Extract all responses
    responses = []
    questions = []
    targets = []
    
    for idx, row in df.iterrows():
        # Extract question
        question = str(row['example']) if pd.notna(row['example']) else ""
        questions.append(question)
        
        # Extract response
        response = extract_text_from_array(row['predictions'])
        responses.append(response)
        
        # Extract target
        target = extract_text_from_array(row['gold'])
        targets.append(target)
    
    # Analyze response lengths
    length_analyses = [analyze_text_length(resp) for resp in responses]
    
    char_lengths = [la['chars'] for la in length_analyses if la['chars'] > 0]
    word_lengths = [la['words'] for la in length_analyses if la['words'] > 0]
    token_lengths = [la['estimated_tokens'] for la in length_analyses if la['estimated_tokens'] > 0]
    
    # Word frequency
    word_freq = get_word_frequency(responses)
    
    analytics = {
        "run_name": run_name,
        "total_samples": len(df),
        "response_length_stats": {
            "characters": {
                "mean": float(np.mean(char_lengths)) if char_lengths else 0,
                "median": float(np.median(char_lengths)) if char_lengths else 0,
                "std": float(np.std(char_lengths)) if char_lengths else 0,
                "min": int(np.min(char_lengths)) if char_lengths else 0,
                "max": int(np.max(char_lengths)) if char_lengths else 0
            },
            "words": {
                "mean": float(np.mean(word_lengths)) if word_lengths else 0,
                "median": float(np.median(word_lengths)) if word_lengths else 0,
                "std": float(np.std(word_lengths)) if word_lengths else 0,
                "min": int(np.min(word_lengths)) if word_lengths else 0,
                "max": int(np.max(word_lengths)) if word_lengths else 0
            },
            "estimated_tokens": {
                "mean": float(np.mean(token_lengths)) if token_lengths else 0,
                "median": float(np.median(token_lengths)) if token_lengths else 0,
                "std": float(np.std(token_lengths)) if token_lengths else 0,
                "min": int(np.min(token_lengths)) if token_lengths else 0,
                "max": int(np.max(token_lengths)) if token_lengths else 0
            }
        },
        "word_frequency": {
            "total_unique_words": len(word_freq),
            "most_common_20": word_freq.most_common(20),
            "frequency_counter": word_freq  # Keep the full counter for comparison
        },
        "questions_and_responses": []
    }
    
    # Add individual question-response pairs
    for i, (question, response, target) in enumerate(zip(questions, responses, targets)):
        item = {
            "index": i,
            "question": question,
            "response": response,
            "target": target,
            "response_analysis": analyze_text_length(response)
        }
        analytics["questions_and_responses"].append(item)
    
    return analytics


def compare_responses(analytics1, analytics2):
    """Compare responses between two runs."""
    responses1 = [item["response"] for item in analytics1["questions_and_responses"]]
    responses2 = [item["response"] for item in analytics2["questions_and_responses"]]
    
    comparison = {
        "identical_responses": 0,
        "different_responses": 0,
        "response_changes": [],
        "length_differences": []
    }
    
    min_len = min(len(responses1), len(responses2))
    
    for i in range(min_len):
        resp1 = responses1[i]
        resp2 = responses2[i]
        
        if resp1 == resp2:
            comparison["identical_responses"] += 1
        else:
            comparison["different_responses"] += 1
            
            len1 = len(resp1) if resp1 else 0
            len2 = len(resp2) if resp2 else 0
            
            comparison["response_changes"].append({
                "index": i,
                "question_preview": analytics1["questions_and_responses"][i]["question"][:100] + "..." if len(analytics1["questions_and_responses"][i]["question"]) > 100 else analytics1["questions_and_responses"][i]["question"],
                "response1_length": len1,
                "response2_length": len2,
                "length_diff": len1 - len2,
                "response1_preview": resp1[:200] + "..." if len(resp1) > 200 else resp1,
                "response2_preview": resp2[:200] + "..." if len(resp2) > 200 else resp2
            })
            
            comparison["length_differences"].append(len1 - len2)
    
    return comparison


def main():
    # File paths
    file1_path = "/Users/admin/Documents/Github/llm-reasoning-token-boost/results/details/Qwen/Qwen2.5-Math-1.5B-Instruct/2025-06-25T10-55-05.197729/details_lighteval|amc23|0_2025-06-25T10-55-05.197729.parquet"
    file2_path = "/Users/admin/Documents/Github/llm-reasoning-token-boost/results/details/Qwen/Qwen2.5-Math-1.5B-Instruct/2025-06-25T11-02-15.155882/details_lighteval|amc23|0_2025-06-25T11-02-15.155882.parquet"
    
    # Load data
    print("Loading evaluation result files...")
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)
    
    print(f"File 1 (Modified Logits): {df1.shape}")
    print(f"File 2 (Baseline): {df2.shape}")
    
    # Analyze both datasets
    analytics1 = analyze_evaluation_data(df1, "Modified_Logits_2025-06-24")
    analytics2 = analyze_evaluation_data(df2, "Baseline_2025-06-18")
    
    # Compare responses
    response_comparison = compare_responses(analytics1, analytics2)
    
    # Compare word frequencies
    word_freq_comparison = compare_word_frequencies(
        analytics1["word_frequency"]["frequency_counter"],
        analytics2["word_frequency"]["frequency_counter"],
        "Modified Logits",
        "Baseline"
    )
    
    # Create comprehensive report
    comparison_report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "model_name": "Qwen2.5-Math-1.5B-Instruct",
        "task": "amc23",
        "runs": {
            "modified_logits": analytics1,
            "baseline": analytics2
        },
        "comparison": response_comparison,
        "word_frequency_comparison": word_freq_comparison,
        "summary": {
            "total_samples": min(analytics1["total_samples"], analytics2["total_samples"]),
            "identical_responses": response_comparison["identical_responses"],
            "different_responses": response_comparison["different_responses"],
            "percentage_different": (response_comparison["different_responses"] / min(analytics1["total_samples"], analytics2["total_samples"])) * 100,
            "response_length_differences": {
                "chars_mean_diff": analytics1["response_length_stats"]["characters"]["mean"] - analytics2["response_length_stats"]["characters"]["mean"],
                "words_mean_diff": analytics1["response_length_stats"]["words"]["mean"] - analytics2["response_length_stats"]["words"]["mean"],
                "tokens_mean_diff": analytics1["response_length_stats"]["estimated_tokens"]["mean"] - analytics2["response_length_stats"]["estimated_tokens"]["mean"]
            }
        }
    }
    
    # Remove the frequency_counter from analytics before saving to avoid JSON serialization issues
    for run_key in comparison_report["runs"]:
        if "frequency_counter" in comparison_report["runs"][run_key]["word_frequency"]:
            del comparison_report["runs"][run_key]["word_frequency"]["frequency_counter"]
    
    # Save to JSON file
    output_file = "evaluation_comparison_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"- Total samples compared: {comparison_report['summary']['total_samples']}")
    print(f"- Identical responses: {response_comparison['identical_responses']}")
    print(f"- Different responses: {response_comparison['different_responses']}")
    print(f"- Percentage different: {comparison_report['summary']['percentage_different']:.1f}%")
    
    print(f"\nResponse Length Statistics:")
    print(f"Modified Logits Run:")
    print(f"  - Avg chars: {analytics1['response_length_stats']['characters']['mean']:.1f}")
    print(f"  - Avg words: {analytics1['response_length_stats']['words']['mean']:.1f}")
    print(f"  - Avg tokens: {analytics1['response_length_stats']['estimated_tokens']['mean']:.1f}")
    
    print(f"Baseline Run:")
    print(f"  - Avg chars: {analytics2['response_length_stats']['characters']['mean']:.1f}")
    print(f"  - Avg words: {analytics2['response_length_stats']['words']['mean']:.1f}")
    print(f"  - Avg tokens: {analytics2['response_length_stats']['estimated_tokens']['mean']:.1f}")
    
    print(f"\nDifferences (Modified - Baseline):")
    print(f"  - Chars: {comparison_report['summary']['response_length_differences']['chars_mean_diff']:.1f}")
    print(f"  - Words: {comparison_report['summary']['response_length_differences']['words_mean_diff']:.1f}")
    print(f"  - Tokens: {comparison_report['summary']['response_length_differences']['tokens_mean_diff']:.1f}")
    
    # Show word frequency differences
    print(f"\n{'='*60}")
    print("WORD FREQUENCY ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nWords appearing MORE in Modified Logits run:")
    for i, item in enumerate(word_freq_comparison['more_frequent_in_run1'][:20]):
        print(f"  {i+1}. '{item['word']}': {item['count_run1']} vs {item['count_run2']} (+{item['absolute_diff']})")
    
    print(f"\nWords appearing MORE in Baseline run:")
    for i, item in enumerate(word_freq_comparison['more_frequent_in_run2'][:20]):
        print(f"  {i+1}. '{item['word']}': {item['count_run2']} vs {item['count_run1']} (+{abs(item['absolute_diff'])})")
    
    print(f"\nMost different words overall (by absolute difference):")
    for i, item in enumerate(word_freq_comparison['most_different_overall'][:20]):
        if item['absolute_diff'] > 0:
            print(f"  {i+1}. '{item['word']}': {item['count_run1']} vs {item['count_run2']} (Modified +{item['absolute_diff']})")
        else:
            print(f"  {i+1}. '{item['word']}': {item['count_run1']} vs {item['count_run2']} (Baseline +{abs(item['absolute_diff'])})")
    
    if analytics1["word_frequency"]["total_unique_words"] > 0:
        print(f"\nTop 5 words in Modified Logits run:")
        for word, count in analytics1["word_frequency"]["most_common_20"][:5]:
            print(f"  - {word}: {count}")
    
    if analytics2["word_frequency"]["total_unique_words"] > 0:
        print(f"\nTop 5 words in Baseline run:")
        for word, count in analytics2["word_frequency"]["most_common_20"][:5]:
            print(f"  - {word}: {count}")


if __name__ == "__main__":
    main() 