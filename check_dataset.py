#!/usr/bin/env python3
"""
Quick script to check where the dataset file is located
"""

import os

def check_dataset_locations():
    print("🔍 Checking dataset locations...")
    print(f"📂 Current working directory: {os.getcwd()}")
    
    # Possible dataset locations
    dataset_locations = [
        "autonomous_thought_data.jsonl",  # Current directory
        os.path.join("data", "autonomous_thought_data.jsonl"),  # Data subdirectory
        os.path.join("..", "autonomous_thought_data.jsonl"),  # Parent directory
        os.path.join("..", "..", "autonomous_thought_data.jsonl"),  # Two levels up
        "/Volumes/Model_Store/Minimum_Consience_AI/autonomous_thought_data.jsonl",  # Absolute path
    ]
    
    found_datasets = []
    
    for i, path in enumerate(dataset_locations, 1):
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        status = "✅ FOUND" if exists else "❌ Missing"
        print(f"{i}. {status}: {abs_path}")
        
        if exists:
            # Check file size
            size = os.path.getsize(abs_path)
            print(f"   📊 Size: {size:,} bytes ({size/1024:.1f} KB)")
            found_datasets.append(abs_path)
    
    print(f"\n📋 Summary:")
    if found_datasets:
        print(f"✅ Found {len(found_datasets)} dataset file(s)")
        print(f"🎯 Recommended: Use the first one found:")
        print(f"   {found_datasets[0]}")
    else:
        print(f"❌ No dataset files found!")
        print(f"💡 Generate dataset first:")
        print(f"   python conscious_ai/autonomus_thinking/create_autonomous_dataset.py")

if __name__ == "__main__":
    check_dataset_locations()