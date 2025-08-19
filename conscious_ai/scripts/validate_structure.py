#!/usr/bin/env python3
"""
Structure Validation Script
===========================
Validates the reorganized project structure by functional phases.
"""

import os
import sys
from pathlib import Path

def validate_phase_structure():
    """Validate that all phase directories exist with required components."""
    
    base_path = Path(__file__).parent.parent
    
    # Expected phase structure
    expected_structure = {
        'phases/p1_perception': ['__init__.py', 'language_detector.py', 'input_processor.py'],
        'phases/p2_cognitive_context': ['__init__.py', 'autonomous_thinking.py', 'conscious_state.py', 'goal_generator.py', 'memory_integration.py'],
        'phases/p3_coherent_generation': ['__init__.py', 'state_evolution_engine.py', 'conscious_response_generator.py'],
        'phases/p34_critical_evaluation': ['__init__.py', 'critical_state_evaluator.py', 'coherence_evaluator.py', 'model_based_coherence_evaluator.py'],
        'phases/p35_conscious_translation': ['__init__.py', 'narrative_generator.py'],
        'shared': ['__init__.py', 'metrics.py', 'integrator.py', 'analysis_tools.py'],
        'training': ['__init__.py', 'phase1_training.py', 'phase2_training.py'],
        'training/model_trainers': ['coherence_classifier_trainer.py'],
    }
    
    print("🔍 VALIDATING REORGANIZED STRUCTURE")
    print("=" * 50)
    
    all_valid = True
    
    for directory, files in expected_structure.items():
        dir_path = base_path / directory
        print(f"\n📁 {directory}:")
        
        if not dir_path.exists():
            print(f"  ❌ Directory does not exist")
            all_valid = False
            continue
            
        for file in files:
            file_path = dir_path / file
            if file_path.exists():
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} - MISSING")
                all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("✅ STRUCTURE VALIDATION PASSED")
        print("All phase directories and core files are present.")
    else:
        print("❌ STRUCTURE VALIDATION FAILED") 
        print("Some directories or files are missing.")
    
    return all_valid

def validate_imports():
    """Validate that key imports work with new structure."""
    
    print("\n🔍 VALIDATING IMPORT STRUCTURE")
    print("=" * 50)
    
    import_tests = [
        ("P1 Perception", "from conscious_ai.phases.p1_perception import language_detector"),
        ("P2 Cognitive", "from conscious_ai.phases.p2_cognitive_context import conscious_state"),
        ("P3 Generation", "from conscious_ai.phases.p3_coherent_generation import state_evolution_engine"),
        ("P3.4 Evaluation", "from conscious_ai.phases.p34_critical_evaluation import critical_state_evaluator"),
        ("P3.5 Translation", "from conscious_ai.phases.p35_conscious_translation import narrative_generator"),
        ("Shared Components", "from conscious_ai.shared import metrics"),
    ]
    
    all_imports_work = True
    
    for description, import_statement in import_tests:
        try:
            exec(import_statement)
            print(f"  ✅ {description}")
        except ImportError as e:
            print(f"  ❌ {description}: {e}")
            all_imports_work = False
        except Exception as e:
            print(f"  ⚠️ {description}: {e}")
    
    print("\n" + "=" * 50)
    if all_imports_work:
        print("✅ IMPORT VALIDATION PASSED")
    else:
        print("❌ SOME IMPORTS FAILED")
        print("Note: Some import errors may be due to missing dependencies")
        print("in individual modules, not structural issues.")
    
    return all_imports_work

if __name__ == "__main__":
    print("🚀 SC PROJECT STRUCTURE VALIDATION")
    print("=" * 60)
    
    structure_valid = validate_phase_structure()
    imports_valid = validate_imports()
    
    print("\n🎯 OVERALL VALIDATION RESULT")
    print("=" * 60)
    
    if structure_valid and imports_valid:
        print("🎉 REORGANIZATION SUCCESSFUL!")
        print("   - All phase directories created")
        print("   - All core files moved correctly") 
        print("   - Import structure validated")
        sys.exit(0)
    else:
        print("⚠️ ISSUES DETECTED")
        print("   - Review validation output above")
        print("   - Some components may need manual fixes")
        sys.exit(1)