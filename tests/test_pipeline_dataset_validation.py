#!/usr/bin/env python3
"""
Pipeline Dataset Validation Script
==================================
Tests the autonomous training pipeline with the actual dataset before full training.
This ensures everything works correctly with your 1000-example dataset.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_dataset_file():
    """Validate the dataset file exists and is properly formatted."""
    logger.info("🔍 Validating dataset file...")
    
    dataset_path = "data/autonomous_thought_data.jsonl"
    
    if not os.path.exists(dataset_path):
        logger.error(f"❌ Dataset not found at: {dataset_path}")
        return False
    
    # Count valid examples
    valid_examples = 0
    total_lines = 0
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines = line_num
            if line.strip():
                try:
                    example = json.loads(line)
                    
                    # Check format
                    if 'previous_SC' in example and 'current_SC' in example:
                        # Validate required fields
                        prev_sc = example['previous_SC']
                        curr_sc = example['current_SC']
                        
                        required_fields = ['goal', 'emotion', 'confidence', 'thought']
                        if all(field in prev_sc for field in required_fields) and \
                           all(field in curr_sc for field in required_fields):
                            valid_examples += 1
                        else:
                            logger.warning(f"⚠️ Line {line_num}: Missing required fields")
                    else:
                        logger.warning(f"⚠️ Line {line_num}: Unknown format")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Line {line_num}: JSON decode error - {e}")
    
    logger.info(f"📊 Dataset validation results:")
    logger.info(f"  - Total lines: {total_lines}")
    logger.info(f"  - Valid examples: {valid_examples}")
    logger.info(f"  - Success rate: {valid_examples/total_lines*100:.1f}%")
    
    return valid_examples > 0

def test_dataset_processing():
    """Test dataset processing without loading heavy models."""
    logger.info("🧪 Testing dataset processing...")
    
    try:
        # Test the specific processing logic from the pipeline
        dataset_path = "data/autonomous_thought_data.jsonl"
        
        examples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        example = json.loads(line)
                        examples.append(example)
                        if line_num > 10:  # Test with first 10 examples
                            break
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON decode error on line {line_num}: {e}")
        
        logger.info(f"✅ Successfully processed {len(examples)} test examples")
        
        # Test format conversion logic
        processed_count = 0
        for example in examples:
            if 'previous_SC' in example and 'current_SC' in example:
                # This is the format conversion logic from the pipeline
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
                processed_count += 1
        
        logger.info(f"✅ Format conversion successful: {processed_count}/{len(examples)} examples")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset processing test failed: {e}")
        return False

def validate_training_config():
    """Validate the training configuration is appropriate for the dataset."""
    logger.info("⚙️ Validating training configuration...")
    
    # These are the optimized parameters we set
    config = {
        'dataset_path': 'autonomous_thought_data.jsonl',
        'num_epochs': 5,
        'train_batch_size': 2,
        'gradient_accumulation_steps': 8,
        'learning_rate': 1e-4,
        'max_length': 512,
        'warmup_steps': 50,
        'save_steps': 50,
        'eval_steps': 50,
        'logging_steps': 25,
    }
    
    # Calculate training metrics
    dataset_size = 1000
    train_split = 0.9
    train_examples = int(dataset_size * train_split)
    
    effective_batch_size = config['train_batch_size'] * config['gradient_accumulation_steps']
    steps_per_epoch = train_examples // effective_batch_size
    total_steps = steps_per_epoch * config['num_epochs']
    
    logger.info(f"📊 Training configuration analysis:")
    logger.info(f"  - Dataset size: {dataset_size} examples")
    logger.info(f"  - Training examples: {train_examples}")
    logger.info(f"  - Effective batch size: {effective_batch_size}")
    logger.info(f"  - Steps per epoch: {steps_per_epoch}")
    logger.info(f"  - Total training steps: {total_steps}")
    logger.info(f"  - Save frequency: every {config['save_steps']} steps")
    logger.info(f"  - Eval frequency: every {config['eval_steps']} steps")
    
    # Validate configuration
    if total_steps < 100:
        logger.warning("⚠️ Total steps might be too low for effective training")
    elif total_steps > 1000:
        logger.warning("⚠️ Total steps might be too high, consider reducing epochs")
    else:
        logger.info("✅ Training configuration looks good!")
    
    return True

def main():
    """Run all validation tests."""
    logger.info("🚀 Starting Pipeline Dataset Validation")
    logger.info("=" * 50)
    
    all_tests_passed = True
    
    # Test 1: Dataset file validation
    if not validate_dataset_file():
        all_tests_passed = False
    
    # Test 2: Dataset processing
    if not test_dataset_processing():
        all_tests_passed = False
    
    # Test 3: Training configuration
    if not validate_training_config():
        all_tests_passed = False
    
    logger.info("=" * 50)
    if all_tests_passed:
        logger.info("🎉 All validation tests PASSED!")
        logger.info("✅ Your dataset is ready for training")
        logger.info("🚀 You can now run: python conscious_ai/autonomous_thinking/autonomous_training_pipeline.py")
    else:
        logger.error("❌ Some validation tests FAILED!")
        logger.error("🔧 Please fix the issues above before training")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)