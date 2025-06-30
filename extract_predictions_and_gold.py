#!/usr/bin/env python3
"""
Extract predicted answers and golden labels from parquet files in the results directory.
Saves the extracted data to JSON format for further analysis.

Usage:
python3 extract_predictions_and_gold.py path/to/specific.parquet output.json
"""

import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime


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


def extract_from_parquet(parquet_file_path):
    """Extract predictions and gold labels from a single parquet file."""
    print(f"Processing: {parquet_file_path}")
    
    try:
        df = pd.read_parquet(parquet_file_path)
        print(f"  Shape: {df.shape}")
        
        extracted_data = []
        
        for idx, row in df.iterrows():
            # Extract basic information
            example = str(row['example']) if pd.notna(row['example']) else ""
            prediction_raw = extract_text_from_array(row['predictions'])
            gold_raw = extract_text_from_array(row['gold'])
            
            # Check if there are extracted predictions in specifics
            extracted_predictions = None
            if pd.notna(row['specifics']) and row['specifics'] is not None:
                specifics = row['specifics']
                if isinstance(specifics, dict) and 'extracted_predictions' in specifics:
                    extracted_predictions = specifics['extracted_predictions']
                    
                    # Convert numpy arrays to lists for JSON serialization
                    if isinstance(extracted_predictions, np.ndarray):
                        extracted_predictions = [
                            arr.tolist() if isinstance(arr, np.ndarray) else arr 
                            for arr in extracted_predictions
                        ]
            
            item = {
                "index": idx,
                "example": example,
                "prediction_raw": prediction_raw,
                "gold_raw": gold_raw,
                "extracted_predictions": extracted_predictions,
                "metrics": row['metrics'] if pd.notna(row['metrics']) else None
            }
            
            extracted_data.append(item)
        
        print(f"  Extracted {len(extracted_data)} items")
        return extracted_data
        
    except Exception as e:
        print(f"  Error processing {parquet_file_path}: {e}")
        return []


def find_parquet_files(results_dir="results/details"):
    """Find all parquet files in the results directory."""
    parquet_files = []
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    return sorted(parquet_files)


def extract_all_parquet_files(output_file="extracted_predictions_and_gold.json"):
    """Extract data from all parquet files and save to JSON."""
    parquet_files = find_parquet_files()
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_extracted_data = {}
    
    for parquet_file in parquet_files:
        # Create a key from the file path
        file_key = parquet_file.replace("results/details/", "").replace(".parquet", "")
        
        # Extract data from this file
        extracted_data = extract_from_parquet(parquet_file)
        
        if extracted_data:
            all_extracted_data[file_key] = {
                "file_path": parquet_file,
                "extraction_timestamp": datetime.now().isoformat(),
                "total_items": len(extracted_data),
                "data": extracted_data
            }
    
    # Save to JSON
    print(f"\nSaving all extracted data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved data from {len(all_extracted_data)} files")
    
    # Print summary statistics
    print("\nSummary:")
    total_items = 0
    files_with_extracted_predictions = 0
    
    for file_key, file_data in all_extracted_data.items():
        items_count = file_data["total_items"]
        total_items += items_count
        
        # Check if any items have extracted predictions
        has_extracted = any(
            item["extracted_predictions"] is not None 
            for item in file_data["data"]
        )
        
        if has_extracted:
            files_with_extracted_predictions += 1
        
        print(f"  {file_key}: {items_count} items {'(with extracted predictions)' if has_extracted else '(raw only)'}")
    
    print(f"\nTotal: {total_items} items from {len(all_extracted_data)} files")
    print(f"Files with extracted predictions: {files_with_extracted_predictions}")
    
    return all_extracted_data


def extract_specific_file(parquet_file_path, output_file=None):
    """Extract data from a specific parquet file."""
    if not os.path.exists(parquet_file_path):
        print(f"Error: File {parquet_file_path} does not exist")
        return None
    
    extracted_data = extract_from_parquet(parquet_file_path)
    
    if not extracted_data:
        print("No data extracted")
        return None
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.basename(parquet_file_path).replace('.parquet', '')
        output_file = f"extracted_{base_name}.json"
    
    # Save to JSON
    output_data = {
        "file_path": parquet_file_path,
        "extraction_timestamp": datetime.now().isoformat(),
        "total_items": len(extracted_data),
        "data": extracted_data
    }
    
    print(f"Saving to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved {len(extracted_data)} items")
    return output_data


def main():
    """Main function with command-line interface."""
    import sys
    
    if len(sys.argv) > 1:
        # Extract specific file
        parquet_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        extract_specific_file(parquet_file, output_file)
    else:
        # Extract all files
        extract_all_parquet_files()


if __name__ == "__main__":
    main()