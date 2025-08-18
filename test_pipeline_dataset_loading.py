#!/usr/bin/env python3
"""
Test script to verify the training pipeline can find and load the dataset correctly
"""

import os
import sys
import tempfile

# Add the project root to the path
sys.path.append('/Volumes/Model_Store/Minimum_Consience_AI')

def test_dataset_loading():
    """Test that the pipeline can find and load the dataset"""
    
    # Change to the autonomus_thinking directory to simulate running from there
    original_cwd = os.getcwd()
    test_dir = '/Volumes/Model_Store/Minimum_Consience_AI/conscious_ai/autonomus_thinking'
    
    try:
        os.chdir(test_dir)
        print(f"📂 Changed to directory: {test_dir}")
        
        # Import the training components
        from autonomous_training_pipeline import TrainingConfig, AutonomousThoughtDataset
        
        # Create a config
        config = TrainingConfig(dataset_path="autonomous_thought_data.jsonl")
        
        # Test dataset search locations
        dataset_locations = [
            os.path.join("data", config.dataset_path),  # Data subdirectory - PRIORITY
            os.path.join(os.path.dirname(__file__), "..", "..", "data", config.dataset_path),  # Repo root data folder
            config.dataset_path,  # Current directory
            os.path.join(os.path.dirname(__file__), "..", "..", config.dataset_path),  # Repo root
            os.path.join("..", "..", config.dataset_path),  # Up two levels
        ]
        
        dataset_found = False
        actual_dataset_path = config.dataset_path
        
        print("🔍 Searching for dataset file...")
        for i, path in enumerate(dataset_locations, 1):
            abs_path = os.path.abspath(path)
            print(f"  {i}. Checking: {abs_path}")
            if os.path.exists(abs_path):
                dataset_found = True
                actual_dataset_path = abs_path
                file_size = os.path.getsize(abs_path)
                print(f"    ✅ FOUND! Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                
                # Quick validation - count lines
                with open(abs_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                print(f"    📄 Contains: {line_count} examples")
                break
            else:
                print(f"    ❌ Not found")
        
        if not dataset_found:
            print("❌ Dataset not found in any location!")
            return False
        
        print(f"\n✅ Dataset found at: {actual_dataset_path}")
        
        # Test loading with the dataset processor
        print("\n🔄 Testing dataset processing...")
        
        # Create a tokenizer mock for testing
        class MockTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "</s>"
                self.pad_token_id = 0
                self.eos_token_id = 1
                
            def encode(self, text, add_special_tokens=False):
                # Simple mock - return token count based on text length
                return list(range(len(text) // 4))  # ~4 chars per token
        
        mock_tokenizer = MockTokenizer()
        dataset_processor = AutonomousThoughtDataset(mock_tokenizer, max_length=512)
        
        # Test loading a small sample
        print("📊 Loading dataset sample...")
        
        # Read just a few examples for testing
        examples = []
        with open(actual_dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Just test first 5 examples
                    break
                if line.strip():
                    import json
                    try:
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON error on line {i+1}: {e}")
                        return False
        
        print(f"📄 Loaded {len(examples)} test examples")
        
        # Test processing
        processed = dataset_processor._process_examples(examples)
        print(f"✅ Successfully processed {len(processed)} examples")
        
        # Check processed format
        if processed:
            sample = processed[0]
            required_keys = ['input_ids', 'attention_mask', 'labels']
            print("🔍 Checking processed example format:")
            for key in required_keys:
                if key in sample:
                    print(f"  ✅ {key}: {len(sample[key])} tokens")
                else:
                    print(f"  ❌ Missing {key}")
                    return False
        
        print("\n🎉 Dataset loading and processing test successful!")
        return True
        
    finally:
        # Restore original directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        print("\n✅ The training pipeline is ready to use the data/ folder dataset!")
    else:
        print("\n❌ There are issues with dataset loading that need to be fixed!")
        sys.exit(1)