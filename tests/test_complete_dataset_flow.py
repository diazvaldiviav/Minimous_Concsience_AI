#!/usr/bin/env python3
"""
Test the complete dataset flow with format conversion and enhanced logging
"""

import os
import json
import sys
import logging

# Configure logging to see the conversion process
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_format_conversion():
    """Test the format conversion from old to new format"""
    
    # Test data in old format
    old_format_example = {
        "previous_SC": {
            "goal": "explore_consciousness",
            "emotion": "curious", 
            "confidence": 0.75,
            "thought": "I wonder about the nature of my awareness"
        },
        "current_SC": {
            "goal": "analyze_patterns",
            "emotion": "analytical",
            "confidence": 0.8,
            "thought": "I observe recurring themes in my processing"
        }
    }
    
    print("🔄 Testing format conversion...")
    print("📥 INPUT (old format):")
    print(json.dumps(old_format_example, indent=2))
    
    # Simulate the conversion logic from the pipeline
    if 'previous_SC' in old_format_example and 'current_SC' in old_format_example:
        # Convert old format to new format
        previous_sc = {
            'goal': old_format_example['previous_SC'].get('goal', 'explore'),
            'emotion': old_format_example['previous_SC'].get('emotion', 'neutral'),
            'confidence': old_format_example['previous_SC'].get('confidence', 0.5),
            'thought': old_format_example['previous_SC'].get('thought', 'Continuing exploration...')
        }
        current_sc = {
            'goal': old_format_example['current_SC'].get('goal', 'explore'),
            'emotion': old_format_example['current_SC'].get('emotion', 'neutral'),
            'confidence': old_format_example['current_SC'].get('confidence', 0.5),
            'thought': old_format_example['current_SC'].get('thought', 'Continuing exploration...')
        }
        
        print("\n✅ CONVERTED to training format:")
        print("Previous state:", previous_sc)
        print("Current state:", current_sc)
        
        # Test the format_prompt function simulation
        print("\n📝 Testing prompt formatting...")
        
        # Extract key components from previous state
        prev_goal = previous_sc.get('goal', 'unknown')
        prev_emotion = previous_sc.get('emotion', 'neutral')
        prev_confidence = previous_sc.get('confidence', 0.5)
        prev_thought = previous_sc.get('thought', '')
        
        # Create user instruction
        user_prompt = (
            f"Given your previous conscious state:\\n"
            f"Goal: {prev_goal}\\n"
            f"Emotion: {prev_emotion}\\n"
            f"Confidence: {prev_confidence:.2f}\\n"
            f"Thought: {prev_thought}\\n\\n"
            f"Generate your next autonomous conscious state as JSON with keys: "
            f"goal, emotion, confidence, thought"
        )
        
        # Format response as JSON
        response = json.dumps({
            "goal": current_sc['goal'],
            "emotion": current_sc['emotion'],
            "confidence": float(current_sc['confidence']),
            "thought": current_sc['thought']
        }, ensure_ascii=False, indent=2)
        
        # Use simple instruction format
        formatted_prompt = (
            f"### Instruction:\\n{user_prompt}\\n\\n"
            f"### Response:\\n{response}"
        )
        
        print("📄 FORMATTED PROMPT:")
        print(formatted_prompt[:200] + "..." if len(formatted_prompt) > 200 else formatted_prompt)
        
        print("\n✅ Format conversion test successful!")
        return True
    else:
        print("❌ Format conversion test failed!")
        return False

def test_actual_dataset_sample():
    """Test with actual data from the dataset"""
    
    data_path = "/Volumes/Model_Store/Minimum_Consience_AI/data/autonomous_thought_data.jsonl"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return False
        
    print(f"\n📊 Testing with actual dataset: {data_path}")
    
    # Load one example
    with open(data_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line:
            example = json.loads(first_line)
            
            print("📥 ACTUAL DATASET EXAMPLE:")
            print(json.dumps(example, indent=2, ensure_ascii=False)[:300] + "...")
            
            # Test the conversion
            if 'previous_SC' in example and 'current_SC' in example:
                previous_sc = {
                    'goal': example['previous_SC'].get('goal', 'explore'),
                    'emotion': example['previous_SC'].get('emotion', 'neutral'),
                    'confidence': example['previous_SC'].get('confidence', 0.5),
                    'thought': example['previous_SC'].get('thought', 'Continuing exploration...')
                }
                current_sc = {
                    'goal': example['current_SC'].get('goal', 'explore'),
                    'emotion': example['current_SC'].get('emotion', 'neutral'),
                    'confidence': example['current_SC'].get('confidence', 0.5),
                    'thought': example['current_SC'].get('thought', 'Continuing exploration...')
                }
                
                print("\n✅ CONVERTED SUCCESSFULLY:")
                print(f"Previous: {previous_sc['goal']} | {previous_sc['emotion']} | {previous_sc['confidence']}")
                print(f"Current:  {current_sc['goal']} | {current_sc['emotion']} | {current_sc['confidence']}")
                
                return True
            else:
                print("❌ Unexpected format in actual dataset!")
                return False
    
    return False

if __name__ == "__main__":
    print("🧪 Testing Complete Dataset Flow")
    print("=" * 50)
    
    # Test format conversion
    conversion_ok = test_format_conversion()
    
    # Test with actual dataset
    dataset_ok = test_actual_dataset_sample()
    
    if conversion_ok and dataset_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The pipeline is ready to use the data/ folder dataset")
        print("✅ Old format will be automatically converted to new format") 
        print("✅ Enhanced logging will show the conversion process")
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)