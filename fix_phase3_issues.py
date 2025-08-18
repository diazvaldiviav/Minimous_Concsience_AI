#!/usr/bin/env python3
"""
Quick fixes for Phase 3 training issues
"""
import os

# Fix 1: Completely disable wandb
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

# Fix 2: Disable tokenizer parallelism warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("✅ Environment variables set for Phase 3 training")
print("Run this before training: python fix_phase3_issues.py")