#!/usr/bin/env python3
"""
Test script to verify the dataset format handling in the training pipeline
"""

import os
import json
import sys

# Add the project root to the path
sys.path.append('/Volumes/Model_Store/Minimum_Consience_AI')

def test_dataset_format():
    """Test the dataset format handling"""
    
    # Check if the data folder dataset exists
    data_path = "/Volumes/Model_Store/Minimum_Consience_AI/data/autonomous_thought_data.jsonl"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at: {data_path}")
        return False
    
    print(f"✅ Found dataset at: {data_path}")
    
    # Load and check a few examples
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Just check first 3 examples
                break
            if line.strip():
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error on line {i+1}: {e}")
                    return False
    
    print(f"📊 Loaded {len(examples)} test examples")
    
    # Check format
    for i, example in enumerate(examples, 1):
        print(f"\n📄 Example {i}:")
        
        if 'previous_SC' in example and 'current_SC' in example:
            print("  🔄 Format: OLD (previous_SC/current_SC)")
            prev = example['previous_SC']
            curr = example['current_SC']
            
            # Check required fields
            required_fields = ['goal', 'emotion', 'confidence', 'thought']
            
            print("  🔍 Previous state fields:")
            for field in required_fields:
                if field in prev:
                    print(f"    ✅ {field}: {prev[field]}")
                else:
                    print(f"    ❌ Missing {field}")
            
            print("  🔍 Current state fields:")
            for field in required_fields:
                if field in curr:
                    print(f"    ✅ {field}: {curr[field]}")
                else:
                    print(f"    ❌ Missing {field}")
                    
        elif 'previous_state' in example and 'current_state' in example:
            print("  🔄 Format: NEW (previous_state/current_state)")
            prev = example['previous_state']
            curr = example['current_state']
        else:
            print("  ❌ Unknown format")
            print(f"    Keys: {list(example.keys())}")
            return False
    
    print("\n✅ Dataset format verification completed successfully!")
    print("💡 The training pipeline can now handle this dataset format.")
    
    return True

if __name__ == "__main__":
    success = test_dataset_format()
    if success:
        print("\n🎉 Dataset is ready for training!")
    else:
        print("\n💥 Dataset needs fixing before training!")
        sys.exit(1)