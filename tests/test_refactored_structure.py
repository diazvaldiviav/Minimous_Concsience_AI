#!/usr/bin/env python3
"""
Test Refactored Structure
========================
Validates that the refactored project structure works correctly.
Tests import paths, configuration system, and core functionality.
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestRefactoredStructure(unittest.TestCase):
    """Test the refactored project structure"""
    
    def test_core_imports(self):
        """Test that core module imports work correctly"""
        try:
            from conscious_ai.core import (
                SensoryModule, ActiveMemory, SelfModel, ReentranceModule,
                CentralIntegrator, ConsciousState, ConsciousStateHistory,
                GoalGenerator, AutomaticThoughtGenerator, ConsciousnessMetrics
            )
            self.assertTrue(True, "Core imports successful")
        except ImportError as e:
            self.fail(f"Core imports failed: {e}")
    
    def test_autonomous_thinking_imports(self):
        """Test autonomous thinking imports with corrected naming"""
        try:
            from conscious_ai.autonomous_thinking.autonomous_thinking import AutonomousThoughtGenerator
            from conscious_ai.autonomous_thinking.autonomous_integration import AutonomousConsciousAI
            self.assertTrue(True, "Autonomous thinking imports successful")
        except ImportError as e:
            self.fail(f"Autonomous thinking imports failed: {e}")
    
    def test_phase34_imports(self):
        """Test Phase 3.4 imports work without duplicates"""
        try:
            from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import CriticalStateEvaluator
            from conscious_ai.coherence_evaluator_model.heuristic_training.coherence_evaluator import CoherenceEvaluator
            self.assertTrue(True, "Phase 3.4 imports successful")
        except ImportError as e:
            self.fail(f"Phase 3.4 imports failed: {e}")
    
    def test_configuration_system(self):
        """Test the new configuration system"""
        try:
            from conscious_ai.config.consciousness_config import (
                get_consciousness_config, get_config_manager, EvaluationMode
            )
            
            # Test config loading
            config = get_consciousness_config()
            self.assertIsNotNone(config)
            self.assertIsNotNone(config.thresholds)
            self.assertIsNotNone(config.phase34)
            
            # Test config manager
            manager = get_config_manager()
            self.assertIsNotNone(manager)
            
            # Test semantic-only mode setting
            manager.set_semantic_only_mode()
            self.assertEqual(config.phase34.evaluation_mode, EvaluationMode.SEMANTIC_ONLY)
            
            print("Configuration system working correctly")
            
        except Exception as e:
            self.fail(f"Configuration system test failed: {e}")
    
    def test_sensitivity_analysis_import(self):
        """Test corrected sensitivity analysis import"""
        try:
            from conscious_ai.modules.sensitivity_analysis import sensibilidad_metricas
            self.assertTrue(callable(sensibilidad_metricas))
            print("Sensitivity analysis import corrected")
        except ImportError as e:
            self.fail(f"Sensitivity analysis import failed: {e}")
    
    def test_main_class_initialization(self):
        """Test that the main class can be initialized with refactored imports"""
        try:
            from conscious_ai.main import MinimalConsciousAI
            
            # This should work without import errors
            ai = MinimalConsciousAI()
            self.assertIsNotNone(ai)
            self.assertIsNotNone(ai.sensory)
            self.assertIsNotNone(ai.memory)
            self.assertIsNotNone(ai.self_model)
            
            print("Main class initialization successful")
            
        except Exception as e:
            self.fail(f"Main class initialization failed: {e}")
    
    def test_no_duplicate_files(self):
        """Test that duplicate files have been removed"""
        project_path = Path(__file__).parent.parent / "conscious_ai"
        
        # These files should no longer exist
        removed_files = [
            "modules/sensory.py",  # Removed duplicate
            "phases/p34_critical_evaluation/coherence_evaluator.py",  # Removed duplicate
            "phases/p34_critical_evaluation/model_based_coherence_evaluator.py",  # Removed duplicate
            "phases/p34_critical_evaluation/fix_consciousness_threshold.py",  # Removed fix file
            "phases/p34_critical_evaluation/quick_fix_phase34.py",  # Removed fix file
            "phases/p34_critical_evaluation/fix_phase34_overfitting.py",  # Removed fix file
            "phases/p3_coherent_generation/fix_phase3_issues.py",  # Removed fix file
        ]
        
        for file_path in removed_files:
            full_path = project_path / file_path
            self.assertFalse(full_path.exists(), f"File should have been removed: {file_path}")
        
        print("Duplicate and fix files properly removed")
    
    def test_directory_naming_fixed(self):
        """Test that directory naming issues have been fixed"""
        project_path = Path(__file__).parent.parent / "conscious_ai"
        
        # Check that autonomous_thinking directory exists (not autonomus_thinking)
        autonomous_path = project_path / "autonomous_thinking"
        self.assertTrue(autonomous_path.exists(), "autonomous_thinking directory should exist")
        
        # Check that old misspelled directory doesn't exist
        old_path = project_path / "autonomus_thinking"
        self.assertFalse(old_path.exists(), "autonomus_thinking directory should not exist")
        
        # Check that sensitivity_analysis.py exists (not sensibilityAnalisys.py)
        sensitivity_path = project_path / "modules" / "sensitivity_analysis.py"
        self.assertTrue(sensitivity_path.exists(), "sensitivity_analysis.py should exist")
        
        print("Directory and file naming corrected")


def run_comprehensive_validation():
    """Run comprehensive validation of the refactored structure"""
    print("REFACTORED STRUCTURE VALIDATION")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRefactoredStructure)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("ALL REFACTORING TESTS PASSED!")
        print("Project structure successfully refactored")
        print("Import paths simplified and working")
        print("Duplicate files removed")
        print("Technical debt addressed")
        print("Configuration system implemented")
        print("\nReady for continued development!")
        return True
    else:
        print("SOME TESTS FAILED!")
        print("Please review the issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)