#!/usr/bin/env python3
"""
Simple validation script for Phase 3.4 fixes - ASCII only
"""

import sys
import os
import re

def check_template_expansion():
    """Check if templates were expanded"""
    print("Checking template expansion...")
    
    file_path = "conscious_ai/coherence_evaluator_model/model_training/coherence_classifier_trainer.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count templates by looking for the 'from' key in each section
        coherent_section = content.split("coherent_transitions = [")[1].split("incoherent_transitions = [")[0]
        coherent_count = coherent_section.count("'from':")
        
        incoherent_section = content.split("incoherent_transitions = [")[1].split("]")[0]
        incoherent_count = incoherent_section.count("'from':")
        
        print(f"Coherent patterns found: {coherent_count}")
        print(f"Incoherent patterns found: {incoherent_count}")
        
        if coherent_count >= 20 and incoherent_count >= 10:
            print("SUCCESS: Templates expanded significantly")
            return True
        else:
            print("WARNING: Template expansion insufficient")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def check_feature_simplification():
    """Check if feature extraction was simplified"""
    print("\nChecking feature simplification...")
    
    file_path = "conscious_ai/coherence_evaluator_model/model_training/coherence_classifier_trainer.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for simplified approach
        if "SIMPLIFIED APPROACH" in content:
            print("SUCCESS: Found simplified approach")
            simplified = True
        else:
            print("WARNING: Simplified approach not found")
            simplified = False
        
        # Check for new methods
        new_methods = [
            "_calculate_goal_similarity",
            "_calculate_emotion_similarity", 
            "_get_basic_embeddings"
        ]
        
        methods_found = sum(1 for method in new_methods if f"def {method}" in content)
        print(f"New methods found: {methods_found}/{len(new_methods)}")
        
        return simplified and methods_found >= 2
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def check_hybrid_optimization():
    """Check if hybrid approach was optimized"""
    print("\nChecking hybrid optimization...")
    
    file_path = "conscious_ai/coherence_evaluator_model/model_training/critical_state_evaluator.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for semantic primary
        semantic_primary = "PRIMARY: Semantic-based evaluator" in content
        ml_secondary = "SECONDARY: ML classifier" in content
        heuristic_fallback = "FALLBACK: Heuristic evaluator" in content
        
        print(f"Semantic as primary: {semantic_primary}")
        print(f"ML as secondary: {ml_secondary}")
        print(f"Heuristic as fallback: {heuristic_fallback}")
        
        if semantic_primary and ml_secondary and heuristic_fallback:
            print("SUCCESS: Hybrid approach optimized")
            return True
        else:
            print("WARNING: Hybrid approach not fully optimized")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Main validation"""
    print("PHASE 3.4 VALIDATION")
    print("=" * 40)
    
    # Run checks
    check1 = check_template_expansion()
    check2 = check_feature_simplification()
    check3 = check_hybrid_optimization()
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"Template expansion: {'PASS' if check1 else 'FAIL'}")
    print(f"Feature simplification: {'PASS' if check2 else 'FAIL'}")
    print(f"Hybrid optimization: {'PASS' if check3 else 'FAIL'}")
    
    total_passed = sum([check1, check2, check3])
    print(f"\nTotal: {total_passed}/3 checks passed")
    
    if total_passed == 3:
        print("\nALL FIXES IMPLEMENTED SUCCESSFULLY!")
        print("Ready for testing with improved model.")
        return 0
    else:
        print(f"\n{3 - total_passed} fixes still need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())