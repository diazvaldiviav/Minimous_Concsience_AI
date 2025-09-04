#!/usr/bin/env python3
"""
Static Test for Phase 6 Memory Consolidation
============================================
Tests the Phase 6 implementation without running the full pipeline.
This ensures all components load correctly and basic functionality works.
"""

import sys
import os
import tempfile
from datetime import datetime
from pathlib import Path

# Add the conscious_ai package to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'conscious_ai'))

def test_phase6_imports():
    """Test that all Phase 6 components can be imported"""
    print("Testing Phase 6 imports...")
    
    try:
        from conscious_ai.phases.p6_memory import (
            MemoryItem, MemoryType, MemoryLayers, MemoryManager, 
            MemoryPersistence, calculate_similarity, find_similar_memories,
            is_personal_info, is_preference, extract_entities,
            merge_memories, abstract_memories
        )
        print("SUCCESS: All Phase 6 imports successful")
        return True
    except ImportError as e:
        print(f"FAILED: Import error: {e}")
        return False

def test_memory_types():
    """Test basic memory data structures"""
    print("Testing memory data structures...")
    
    try:
        from conscious_ai.phases.p6_memory.memory_types import MemoryItem, MemoryType, MemoryLayers
        
        # Test MemoryItem creation
        memory = MemoryItem(
            content="Test memory content",
            relevance=0.8,
            timestamp=datetime.now(),
            memory_type=MemoryType.WORKING
        )
        
        assert memory.content == "Test memory content"
        assert memory.relevance == 0.8
        assert memory.memory_type == MemoryType.WORKING
        
        # Test decay and access
        initial_relevance = memory.relevance
        memory.decay()
        assert memory.relevance < initial_relevance
        
        memory.access()
        assert memory.access_count == 1
        
        # Test MemoryLayers
        layers = MemoryLayers()
        success = layers.add_to_working(memory)
        assert success == True
        assert len(layers.working_memory) == 1
        
        print("SUCCESS: Memory data structures working correctly")
        return True
        
    except Exception as e:
        print(f"FAILED: Memory types test failed: {e}")
        return False

def test_memory_utils():
    """Test memory utility functions"""
    print("Testing memory utilities...")
    
    try:
        from conscious_ai.phases.p6_memory.memory_utils import (
            is_personal_info, is_preference, extract_entities
        )
        
        # Test personal info detection
        personal_tests = [
            ("My name is John", True),
            ("I am a software engineer", True),
            ("Call me Jane", True),
            ("What is the weather", False),
            ("I like pizza", False)
        ]
        
        for text, expected in personal_tests:
            result = is_personal_info(text)
            assert result == expected, f"Personal info test failed for '{text}': got {result}, expected {expected}"
        
        # Test preference detection
        preference_tests = [
            ("I like classical music", True),
            ("I prefer tea over coffee", True),
            ("I think this is good", True),
            ("The weather is nice", False),
            ("What time is it", False)
        ]
        
        for text, expected in preference_tests:
            result = is_preference(text)
            assert result == expected, f"Preference test failed for '{text}': got {result}, expected {expected}"
        
        # Test entity extraction
        entities = extract_entities("My name is John Smith and I work at Google")
        assert len(entities) > 0
        
        print("SUCCESS: Memory utilities working correctly")
        return True
        
    except Exception as e:
        print(f"FAILED: Memory utils test failed: {e}")
        return False

def test_memory_manager():
    """Test memory manager functionality"""
    print("Testing memory manager...")
    
    try:
        from conscious_ai.phases.p6_memory.memory_manager import MemoryManager
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize memory manager with temp path
            manager = MemoryManager(auto_persist=False)  # Disable auto-persist for testing
            
            # Test adding memories
            mem1 = manager.add_memory("My name is Alice", relevance=0.9, auto_consolidate=False)
            assert mem1.content == "My name is Alice"
            assert mem1.metadata.get('is_personal') == True
            
            mem2 = manager.add_memory("I love chocolate ice cream", relevance=0.7, auto_consolidate=False)
            assert mem2.metadata.get('is_preference') == True
            
            # Test memory retrieval
            results = manager.retrieve_relevant("What is my name?", top_k=3)
            assert len(results) > 0
            found_alice = any("alice" in mem.content.lower() for mem, score in results)
            assert found_alice, "Should find Alice in personal info"
            
            # Test statistics
            stats = manager.get_statistics()
            assert stats['working_memory']['count'] >= 2
            assert stats['total_memories'] >= 2
            
            # Test consolidation trigger
            for i in range(20):
                manager.add_memory(f"Test memory {i}", relevance=0.3, auto_consolidate=False)
            
            should_consolidate = manager.should_consolidate()
            assert should_consolidate == True
            
            # Test consolidation
            consolidation_stats = manager.consolidate()
            assert 'duration_ms' in consolidation_stats
            assert consolidation_stats['duration_ms'] < 1000  # Should be fast
            
        print("SUCCESS: Memory manager working correctly")
        return True
        
    except Exception as e:
        print(f"FAILED: Memory manager test failed: {e}")
        return False

def test_persistence():
    """Test JSON persistence functionality"""
    print("Testing persistence...")
    
    try:
        from conscious_ai.phases.p6_memory.persistence import MemoryPersistence
        from conscious_ai.phases.p6_memory.memory_types import MemoryLayers, MemoryItem, MemoryType
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_memory.json"
            
            # Create test memory layers
            layers = MemoryLayers()
            
            # Add test memories
            mem1 = MemoryItem("Test memory 1", 0.8, datetime.now(), memory_type=MemoryType.WORKING)
            mem2 = MemoryItem("Test memory 2", 0.6, datetime.now(), memory_type=MemoryType.EPISODIC)
            mem3 = MemoryItem("Important fact", 0.9, datetime.now(), memory_type=MemoryType.CORE)
            
            layers.add_to_working(mem1)
            layers.add_to_episodic(mem2)
            layers.add_to_core(mem3)
            
            # Test saving
            persistence = MemoryPersistence(str(temp_dir))
            success = persistence.save_to_json(layers, test_file)
            assert success == True
            assert test_file.exists()
            
            # Test loading
            loaded_layers = persistence.load_from_json(test_file)
            assert loaded_layers is not None
            assert len(loaded_layers.working_memory) == 1
            assert len(loaded_layers.episodic_buffer) == 1
            assert len(loaded_layers.core_knowledge) == 1
            
            # Verify content
            assert loaded_layers.working_memory[0].content == "Test memory 1"
            assert loaded_layers.episodic_buffer[0].content == "Test memory 2"
            assert loaded_layers.core_knowledge[0].content == "Important fact"
        
        print("SUCCESS: Persistence working correctly")
        return True
        
    except Exception as e:
        print(f"FAILED: Persistence test failed: {e}")
        return False

def test_pipeline_integration():
    """Test integration with pipeline orchestrator"""
    print("Testing pipeline integration...")
    
    try:
        from conscious_ai.core.pipeline_orchestrator import ConsciousnessPipelineOrchestrator
        
        # Initialize orchestrator
        orchestrator = ConsciousnessPipelineOrchestrator(enable_phase4=False, debug=False)
        
        # Check Phase 6 availability
        assert hasattr(orchestrator, 'phase6_available')
        assert hasattr(orchestrator, 'memory_manager')
        
        if orchestrator.phase6_available:
            assert orchestrator.memory_manager is not None
            print("SUCCESS: Phase 6 successfully integrated into pipeline")
        else:
            print("WARNING: Phase 6 not available in pipeline (expected if dependencies missing)")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Pipeline integration test failed: {e}")
        return False

def main():
    """Run all Phase 6 static tests"""
    print("Phase 6 Memory Consolidation - Static Test Suite")
    print("=" * 60)
    
    tests = [
        test_phase6_imports,
        test_memory_types,
        test_memory_utils,
        test_memory_manager,
        test_persistence,
        test_pipeline_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"FAILED: Test {test.__name__} crashed: {e}")
        print()
    
    print("Test Results")
    print("-" * 20)
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("SUCCESS: All tests passed! Phase 6 implementation is ready.")
        return True
    else:
        print("FAILED: Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)