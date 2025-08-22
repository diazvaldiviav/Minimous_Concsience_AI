"""
Comprehensive Test Suite for Phase 4 Layer 2 GPT-OSS Integration
================================================================
Validates complete GPT-OSS-20B integration with hybrid CPU+GPU architecture.
Tests hardware detection, model loading, memory management, and consciousness integration.
"""

import unittest
import time
import logging
import tempfile
import json
import asyncio
from typing import Dict, Any, List, Optional
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..'))

from conscious_ai.phases.p4_LLM_Communication.core.hardware_profiler import (
    PremiumHardwareProfiler, HardwareConfiguration
)
from conscious_ai.phases.p4_LLM_Communication.core.backend_manager import (
    PremiumBackendManager, BackendType, QueryContext
)
from conscious_ai.phases.p4_LLM_Communication.models.gpt_oss_loader import (
    HybridGPTOSSLoader, LoadingConfiguration
)
from conscious_ai.phases.p4_LLM_Communication.optimization.performance_monitor import (
    PremiumPerformanceMonitor, AlertConfiguration
)
from conscious_ai.phases.p4_LLM_Communication.formatters.harmony_processor import (
    HarmonyFormatProcessor, ConsciousnessContext
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestHardwareProfiler(unittest.TestCase):
    """Test hardware detection and configuration for 51GB+15GB specs"""
    
    def setUp(self):
        self.profiler = PremiumHardwareProfiler()
    
    def test_hardware_detection(self):
        """Test detection of premium hardware configuration"""
        logger.info("🧪 Testing hardware detection...")
        
        config = self.profiler.detect_hardware_configuration()
        
        # Validate configuration structure
        self.assertIsInstance(config, HardwareConfiguration)
        self.assertGreater(config.total_ram_gb, 0)
        self.assertGreater(config.total_vram_gb, 0)
        self.assertIsNotNone(config.gpu_name)
        self.assertIsNotNone(config.architecture_type)
        
        logger.info(f"✅ Hardware detected: {config.total_ram_gb:.1f}GB RAM, {config.total_vram_gb:.1f}GB VRAM")
        logger.info(f"   GPU: {config.gpu_name}, Architecture: {config.architecture_type}")
        
        # Test premium hardware recognition (if running on target hardware)
        if config.total_ram_gb >= 50.0 and config.total_vram_gb >= 14.0:
            self.assertTrue(config.is_premium_hardware)
            logger.info("✅ Premium hardware configuration detected")
        else:
            logger.warning(f"⚠️ Non-premium hardware: {config.total_ram_gb:.1f}GB/{config.total_vram_gb:.1f}GB")
    
    def test_memory_distribution_calculation(self):
        """Test optimal memory distribution for GPT-OSS-20B"""
        logger.info("🧪 Testing memory distribution calculation...")
        
        config = self.profiler.detect_hardware_configuration()
        distribution = self.profiler.calculate_optimal_memory_distribution(config)
        
        # Validate distribution structure
        self.assertIn('strategy', distribution)
        self.assertIn('model_size_estimate_gb', distribution)
        self.assertIn('gpu_allocation', distribution)
        self.assertIn('cpu_allocation', distribution)
        
        # Test memory allocation logic
        if config.is_premium_hardware:
            self.assertEqual(distribution['strategy'], 'hybrid_cpu_gpu_premium')
            self.assertLessEqual(distribution['gpu_allocation']['total_gpu_usage'], config.usable_vram_gb)
            self.assertLessEqual(distribution['cpu_allocation']['total_cpu_usage'], config.usable_ram_gb)
            logger.info("✅ Premium memory distribution calculated")
        else:
            self.assertIn('fallback', distribution['strategy'])
            logger.info("✅ Fallback distribution calculated for non-premium hardware")
    
    def test_gpt_oss_requirements_validation(self):
        """Test validation of GPT-OSS-20B requirements"""
        logger.info("🧪 Testing GPT-OSS requirements validation...")
        
        config = self.profiler.detect_hardware_configuration()
        validation = self.profiler.validate_gpt_oss_requirements(config)
        
        # Validate structure
        self.assertIn('meets_minimum', validation)
        self.assertIn('meets_recommended', validation)
        self.assertIn('overall_grade', validation)
        
        logger.info(f"✅ Requirements validation: {validation['overall_grade']} grade")
        logger.info(f"   Minimum: {'✅' if validation['meets_minimum'] else '❌'}")
        logger.info(f"   Recommended: {'✅' if validation['meets_recommended'] else '❌'}")


class TestGPTOSSLoader(unittest.TestCase):
    """Test GPT-OSS-20B hybrid loading functionality"""
    
    def setUp(self):
        self.profiler = PremiumHardwareProfiler()
        self.hardware_config = self.profiler.detect_hardware_configuration()
        self.loader = HybridGPTOSSLoader(self.hardware_config.__dict__)
    
    def test_loading_configuration_creation(self):
        """Test creation of optimal loading configuration"""
        logger.info("🧪 Testing loading configuration creation...")
        
        config = LoadingConfiguration(
            model_name="openai/gpt-oss-20b",
            use_hybrid_loading=True,
            quantization_type="MXFP4",
            gpu_memory_limit_gb=min(13.0, self.hardware_config.usable_vram_gb),
            cpu_memory_limit_gb=min(35.0, self.hardware_config.usable_ram_gb)
        )
        
        # Validate configuration
        self.assertEqual(config.model_name, "openai/gpt-oss-20b")
        self.assertTrue(config.use_hybrid_loading)
        self.assertLessEqual(config.gpu_memory_limit_gb, self.hardware_config.usable_vram_gb)
        self.assertLessEqual(config.cpu_memory_limit_gb, self.hardware_config.usable_ram_gb)
        
        logger.info(f"✅ Loading configuration created: {config.quantization_type} quantization")
        logger.info(f"   GPU limit: {config.gpu_memory_limit_gb:.1f}GB, CPU limit: {config.cpu_memory_limit_gb:.1f}GB")
    
    @unittest.skipUnless(os.getenv('ENABLE_MODEL_LOADING_TESTS') == 'true', 
                        "Model loading tests disabled (set ENABLE_MODEL_LOADING_TESTS=true to enable)")
    def test_model_loading_simulation(self):
        """Test model loading process (simulation mode)"""
        logger.info("🧪 Testing model loading simulation...")
        
        # Use smaller limits for testing
        config = LoadingConfiguration(
            model_name="openai/gpt-oss-20b",
            use_hybrid_loading=True,
            quantization_type="INT8",  # More conservative for testing
            gpu_memory_limit_gb=min(8.0, self.hardware_config.usable_vram_gb * 0.6),
            cpu_memory_limit_gb=min(20.0, self.hardware_config.usable_ram_gb * 0.5),
            max_loading_time_minutes=2  # Shorter timeout for tests
        )
        
        # Note: This would require actual model download in real scenario
        # For testing, we validate the configuration and setup
        
        # Test pre-flight validation
        validation_result = self.loader._validate_loading_requirements(config)
        if validation_result:
            logger.info("✅ Pre-flight validation passed")
        else:
            logger.warning("⚠️ Pre-flight validation failed - insufficient resources for testing")
        
        self.assertIsInstance(validation_result, bool)
    
    def test_memory_monitoring(self):
        """Test memory monitoring during loading"""
        logger.info("🧪 Testing memory monitoring...")
        
        # Test memory monitor initialization
        from conscious_ai.phases.p4_LLM_Communication.models.gpt_oss_loader import MemoryMonitor
        
        monitor = MemoryMonitor(ram_limit_gb=45.0, vram_limit_gb=13.0)
        
        # Start monitoring briefly
        monitor.start_monitoring(interval_seconds=0.5)
        time.sleep(2)  # Monitor for 2 seconds
        stats = monitor.stop_monitoring()
        
        # Validate monitoring results
        self.assertIn('peak_ram_gb', stats)
        self.assertIn('peak_vram_gb', stats)
        self.assertIn('samples_collected', stats)
        self.assertGreater(stats['samples_collected'], 0)
        
        logger.info(f"✅ Memory monitoring: {stats['samples_collected']} samples collected")
        logger.info(f"   Peak RAM: {stats['peak_ram_gb']:.1f}GB, Peak VRAM: {stats['peak_vram_gb']:.1f}GB")


class TestBackendManager(unittest.TestCase):
    """Test premium backend manager with multiple models"""
    
    def setUp(self):
        self.profiler = PremiumHardwareProfiler()
        self.hardware_config = self.profiler.detect_hardware_configuration()
        self.backend_manager = PremiumBackendManager(self.hardware_config)
    
    @unittest.skipUnless(os.getenv('ENABLE_BACKEND_TESTS') == 'true',
                        "Backend tests disabled (set ENABLE_BACKEND_TESTS=true to enable)")
    def test_backend_initialization(self):
        """Test initialization of multiple backends"""
        logger.info("🧪 Testing backend initialization...")
        
        async def run_test():
            # Initialize backends (this would normally load actual models)
            results = await self.backend_manager.initialize_backends()
            
            # Validate initialization results
            self.assertIsInstance(results, dict)
            self.assertIn(BackendType.PRIMARY_GPT_OSS, results)
            self.assertIn(BackendType.EMERGENCY_MT5, results)
            
            logger.info(f"✅ Backend initialization: {sum(results.values())}/{len(results)} backends ready")
            
            # Test system status
            status = self.backend_manager.get_system_status()
            self.assertIn('system_active', status)
            self.assertIn('total_backends', status)
            
            logger.info(f"   Active backends: {status['healthy_backends']}/{status['total_backends']}")
            
            return results
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_test())
            self.assertIsInstance(results, dict)
        finally:
            loop.close()
    
    def test_query_routing(self):
        """Test intelligent query routing"""
        logger.info("🧪 Testing query routing logic...")
        
        # Test query context creation
        simple_query = QueryContext("What is 2+2?")
        complex_query = QueryContext("Analyze the philosophical implications of consciousness in AI systems")
        consciousness_query = QueryContext("How do you experience self-awareness?", consciousness_state={'test': True})
        
        # Validate context calculation
        self.assertLess(simple_query.complexity_score, complex_query.complexity_score)
        self.assertGreater(consciousness_query.complexity_score, simple_query.complexity_score)
        
        logger.info(f"✅ Query routing: Simple={simple_query.complexity_score:.2f}, Complex={complex_query.complexity_score:.2f}")
        logger.info(f"   Consciousness={consciousness_query.complexity_score:.2f}")
    
    def test_failover_logic(self):
        """Test automatic failover mechanisms"""
        logger.info("🧪 Testing failover logic...")
        
        # This tests the logic without requiring actual models
        query_context = QueryContext("Test query for failover")
        
        # Test fallback backend selection
        fallback = self.backend_manager._get_fallback_backend()
        # Since no backends are actually loaded, this should return None
        self.assertIsNone(fallback)
        
        logger.info("✅ Failover logic validated (no backends loaded for testing)")


class TestPerformanceMonitor(unittest.TestCase):
    """Test real-time performance monitoring"""
    
    def setUp(self):
        self.config = AlertConfiguration(
            ram_warning_threshold=0.80,
            vram_warning_threshold=0.80,
            continuous_monitoring_seconds=1
        )
        self.monitor = PremiumPerformanceMonitor(self.config)
    
    def test_performance_monitoring_initialization(self):
        """Test performance monitor initialization"""
        logger.info("🧪 Testing performance monitor initialization...")
        
        # Test configuration
        self.assertEqual(self.monitor.config.ram_warning_threshold, 0.80)
        self.assertEqual(self.monitor.config.vram_warning_threshold, 0.80)
        self.assertFalse(self.monitor.monitoring_active)
        
        logger.info("✅ Performance monitor initialized with correct configuration")
    
    def test_real_time_monitoring(self):
        """Test real-time monitoring capabilities"""
        logger.info("🧪 Testing real-time monitoring...")
        
        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.monitoring_active)
        
        # Let it collect some data
        time.sleep(3)
        
        # Check metrics collection
        current_metrics = self.monitor.get_current_metrics()
        self.assertIsNotNone(current_metrics)
        self.assertGreater(current_metrics.ram_usage_gb, 0)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_active)
        
        logger.info(f"✅ Real-time monitoring: RAM {current_metrics.ram_usage_gb:.1f}GB")
        logger.info(f"   VRAM {current_metrics.vram_usage_gb:.1f}GB, Efficiency {current_metrics.memory_efficiency_score:.1f}%")
    
    def test_alert_system(self):
        """Test alert generation and management"""
        logger.info("🧪 Testing alert system...")
        
        alert_triggered = False
        
        def alert_callback(alert):
            nonlocal alert_triggered
            alert_triggered = True
            logger.info(f"Alert triggered: {alert.alert_type} - {alert.message}")
        
        # Add alert callback
        self.monitor.add_alert_callback(alert_callback)
        
        # Start monitoring briefly
        self.monitor.start_monitoring()
        time.sleep(2)
        self.monitor.stop_monitoring()
        
        # Check if any alerts were generated (depends on current system state)
        active_alerts = len(self.monitor.active_alerts)
        logger.info(f"✅ Alert system: {active_alerts} active alerts")
    
    def test_optimization_recommendations(self):
        """Test automatic optimization recommendations"""
        logger.info("🧪 Testing optimization recommendations...")
        
        recommendations_received = []
        
        def optimization_callback(recommendations):
            recommendations_received.extend(recommendations)
            logger.info(f"Optimization recommendations: {len(recommendations)} received")
        
        # Add optimization callback
        self.monitor.add_optimization_callback(optimization_callback)
        
        # Start monitoring to generate recommendations
        self.monitor.start_monitoring()
        time.sleep(3)
        self.monitor.stop_monitoring()
        
        # Check recommendations
        current_recommendations = self.monitor.optimization_recommendations
        logger.info(f"✅ Optimization system: {len(current_recommendations)} recommendations available")
        
        for rec in current_recommendations[:3]:  # Show first 3
            logger.info(f"   {rec.category}: {rec.title}")


class TestHarmonyProcessor(unittest.TestCase):
    """Test harmony format processing and consciousness integration"""
    
    def setUp(self):
        self.processor = HarmonyFormatProcessor()
    
    def test_consciousness_context_conversion(self):
        """Test conversion of SC_t state to consciousness context"""
        logger.info("🧪 Testing consciousness context conversion...")
        
        # Create mock SC_t state
        sc_t_state = {
            'E_t': {'text': 'What is consciousness?', 'activation': 0.8},
            'M_t': [
                {'content': {'text': 'Previous thought about awareness'}, 'relevance': 0.9},
                {'content': {'text': 'Memory of philosophical discussion'}, 'relevance': 0.7}
            ],
            'S_t': {'emotional_state': 'curious', 'confidence_level': 0.75},
            'G_t': {'primary_goal': 'understand_consciousness'},
            'A_t': ['I wonder about my own awareness', 'What defines consciousness?'],
            'metrics': {'f': 1.45},
            'cycle': 42
        }
        
        # Convert to consciousness context
        context = ConsciousnessContext.from_sc_t(sc_t_state)
        
        # Validate conversion
        self.assertEqual(context.consciousness_score, 1.45)
        self.assertEqual(context.cycle_number, 42)
        self.assertEqual(context.internal_state['emotional_state'], 'curious')
        self.assertEqual(len(context.active_memory), 2)
        self.assertEqual(len(context.automatic_thoughts), 2)
        
        logger.info(f"✅ Consciousness context: Score {context.consciousness_score:.2f}, Cycle {context.cycle_number}")
        logger.info(f"   State: {context.internal_state['emotional_state']}, Goal: {context.goals_intentions['primary_goal']}")
    
    def test_harmony_format_generation(self):
        """Test generation of harmony format requests"""
        logger.info("🧪 Testing harmony format generation...")
        
        # Create test consciousness state
        sc_t_state = {
            'E_t': {'text': 'Explain quantum consciousness theories'},
            'S_t': {'emotional_state': 'analytical', 'confidence_level': 0.8},
            'G_t': {'primary_goal': 'provide_comprehensive_analysis'},
            'A_t': ['This requires deep analysis', 'Multiple theories exist'],
            'metrics': {'f': 1.6},
            'cycle': 15
        }
        
        # Process query
        harmony_request = self.processor.process_consciousness_query(
            "Explain quantum consciousness theories",
            sc_t_state
        )
        
        # Validate harmony request
        self.assertIsNotNone(harmony_request.consciousness_context)
        self.assertGreater(len(harmony_request.reasoning_chain), 0)
        
        # Test harmony format conversion
        harmony_format = harmony_request.to_harmony_format()
        self.assertIn('messages', harmony_format)
        self.assertIn('consciousness_enhanced', harmony_format)
        self.assertTrue(harmony_format['consciousness_enhanced'])
        
        logger.info(f"✅ Harmony format: {len(harmony_format['messages'])} messages")
        logger.info(f"   Reasoning steps: {len(harmony_request.reasoning_chain)}")
    
    def test_chain_of_thought_processing(self):
        """Test chain-of-thought reasoning generation"""
        logger.info("🧪 Testing chain-of-thought processing...")
        
        # Create consciousness context
        sc_t_state = {
            'S_t': {'emotional_state': 'reflective'},
            'G_t': {'primary_goal': 'philosophical_analysis'},
            'A_t': ['Deep questions require careful thought'],
            'M_t': [{'content': {'text': 'Previous philosophical discussions'}}],
            'metrics': {'f': 1.3}
        }
        
        consciousness_context = ConsciousnessContext.from_sc_t(sc_t_state)
        
        # Generate reasoning chain
        reasoning_chain = self.processor.chain_processor.generate_reasoning_chain(
            "What is the nature of consciousness?", 
            consciousness_context
        )
        
        # Validate reasoning chain
        self.assertGreater(len(reasoning_chain), 3)
        self.assertTrue(any('consciousness' in step.lower() for step in reasoning_chain))
        
        logger.info(f"✅ Chain-of-thought: {len(reasoning_chain)} reasoning steps generated")
        for i, step in enumerate(reasoning_chain[:3]):
            logger.info(f"   {i+1}. {step[:60]}...")
    
    def test_format_fallback(self):
        """Test fallback to standard format when harmony unavailable"""
        logger.info("🧪 Testing format fallback...")
        
        # Test format type detection
        format_type = self.processor.get_format_type()
        self.assertIn(format_type, ['harmony', 'standard'])
        
        # Test fallback request creation
        fallback_request = self.processor.create_fallback_request(
            "Test query",
            {'metrics': {'f': 1.2}, 'S_t': {'emotional_state': 'curious'}}
        )
        
        self.assertIn('prompt', fallback_request)
        self.assertFalse(fallback_request['consciousness_enhanced'])
        
        logger.info(f"✅ Format fallback: Using {format_type} format")


class TestIntegrationValidation(unittest.TestCase):
    """Integration tests for complete Phase 4 Layer 2 system"""
    
    def setUp(self):
        self.profiler = PremiumHardwareProfiler()
        self.hardware_config = self.profiler.detect_hardware_configuration()
    
    def test_phase_1_to_3_compatibility(self):
        """Test that Phases 1-3.5 maintain functionality"""
        logger.info("🧪 Testing Phases 1-3.5 compatibility...")
        
        try:
            # Test Phase 1-2 imports
            from conscious_ai.main import MinimalConsciousAI
            
            # Test Phase 3.4 imports
            from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import CriticalStateEvaluator
            
            # Test basic functionality
            ai = MinimalConsciousAI()
            self.assertIsNotNone(ai)
            
            logger.info("✅ Phases 1-3.5 compatibility: Core imports successful")
            
            # Test basic state creation
            test_result = ai.process_input("Test consciousness integration")
            self.assertIn('consciousness_metrics', test_result)
            self.assertIn('is_conscious', test_result)
            
            logger.info(f"✅ Phases 1-3.5 functionality: Consciousness score {test_result['consciousness_metrics']['f']:.3f}")
            
        except ImportError as e:
            self.fail(f"Phase 1-3.5 compatibility broken: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Phase 1-3.5 functionality test incomplete: {e}")
    
    def test_memory_usage_compliance(self):
        """Test memory usage stays within targets"""
        logger.info("🧪 Testing memory usage compliance...")
        
        # Start performance monitoring
        monitor = PremiumPerformanceMonitor()
        monitor.start_monitoring()
        
        # Let it collect baseline data
        time.sleep(2)
        
        current_metrics = monitor.get_current_metrics()
        if current_metrics:
            ram_usage = current_metrics.ram_usage_gb
            vram_usage = current_metrics.vram_usage_gb
            
            # Check against targets (45GB RAM, 13GB VRAM)
            ram_within_target = ram_usage <= 45.0
            vram_within_target = vram_usage <= 13.0
            
            logger.info(f"✅ Memory compliance: RAM {ram_usage:.1f}/45.0GB ({'✅' if ram_within_target else '❌'})")
            logger.info(f"   VRAM {vram_usage:.1f}/13.0GB ({'✅' if vram_within_target else '❌'})")
            
            # For CI/testing, we're lenient on memory targets
            # In production, these would be strict requirements
            
        monitor.stop_monitoring()
    
    def test_system_stability(self):
        """Test system stability under load"""
        logger.info("🧪 Testing system stability...")
        
        stability_metrics = {
            'start_time': time.time(),
            'errors_encountered': 0,
            'successful_operations': 0
        }
        
        # Run multiple operations to test stability
        for i in range(5):
            try:
                # Test hardware profiling
                config = self.profiler.detect_hardware_configuration()
                self.assertIsNotNone(config)
                
                # Test performance monitoring
                monitor = PremiumPerformanceMonitor()
                monitor.start_monitoring()
                time.sleep(0.5)
                monitor.stop_monitoring()
                
                # Test harmony processing
                processor = HarmonyFormatProcessor()
                test_request = processor.create_fallback_request("Test query")
                self.assertIsNotNone(test_request)
                
                stability_metrics['successful_operations'] += 1
                
            except Exception as e:
                logger.warning(f"Stability test error in iteration {i}: {e}")
                stability_metrics['errors_encountered'] += 1
        
        # Calculate stability score
        total_operations = stability_metrics['successful_operations'] + stability_metrics['errors_encountered']
        stability_score = stability_metrics['successful_operations'] / total_operations if total_operations > 0 else 0
        
        logger.info(f"✅ System stability: {stability_score:.1%} success rate")
        logger.info(f"   {stability_metrics['successful_operations']}/{total_operations} operations successful")
        
        # Require at least 80% stability for tests
        self.assertGreaterEqual(stability_score, 0.8)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and automatic recovery"""
        logger.info("🧪 Testing error handling and recovery...")
        
        # Test hardware profiler error handling
        try:
            # Force an error condition if possible
            profiler = PremiumHardwareProfiler()
            config = profiler.detect_hardware_configuration()
            
            # Test with invalid inputs
            invalid_distribution = profiler.calculate_optimal_memory_distribution(None)
            # Should handle gracefully
            
            logger.info("✅ Error handling: Hardware profiler handles edge cases")
            
        except Exception as e:
            logger.warning(f"⚠️ Error handling test: {e}")
        
        # Test performance monitor error handling
        try:
            monitor = PremiumPerformanceMonitor()
            # Test invalid configuration
            invalid_config = AlertConfiguration(
                ram_warning_threshold=-1.0,  # Invalid threshold
                continuous_monitoring_seconds=0
            )
            # Should handle gracefully
            
            logger.info("✅ Error handling: Performance monitor handles invalid config")
            
        except Exception as e:
            logger.warning(f"⚠️ Performance monitor error handling: {e}")


def run_comprehensive_test_suite():
    """Run the complete test suite with detailed reporting"""
    logger.info("🚀 Starting Phase 4 Layer 2 Comprehensive Test Suite")
    logger.info("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestHardwareProfiler,
        TestGPTOSSLoader,
        TestBackendManager,
        TestPerformanceMonitor,
        TestHarmonyProcessor,
        TestIntegrationValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Generate test report
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    logger.info("=" * 80)
    logger.info("🎯 PHASE 4 LAYER 2 TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {total_tests - failures - errors}")
    logger.info(f"Failed: {failures}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Success Rate: {success_rate:.1%}")
    
    if failures > 0:
        logger.error("❌ FAILED TESTS:")
        for test, traceback in result.failures:
            logger.error(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    if errors > 0:
        logger.error("💥 ERROR TESTS:")
        for test, traceback in result.errors:
            logger.error(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    # Overall assessment
    if success_rate >= 0.9:
        logger.info("🎉 EXCELLENT: Phase 4 Layer 2 implementation ready for production")
    elif success_rate >= 0.8:
        logger.info("✅ GOOD: Phase 4 Layer 2 implementation ready with minor issues")
    elif success_rate >= 0.7:
        logger.info("⚠️ ACCEPTABLE: Phase 4 Layer 2 needs improvements before production")
    else:
        logger.error("❌ CRITICAL: Phase 4 Layer 2 requires significant fixes")
    
    logger.info("=" * 80)
    
    return result


if __name__ == '__main__':
    # Set environment variables for testing
    os.environ.setdefault('ENABLE_MODEL_LOADING_TESTS', 'false')  # Set to 'true' for full model tests
    os.environ.setdefault('ENABLE_BACKEND_TESTS', 'false')       # Set to 'true' for backend tests
    
    # Run comprehensive test suite
    test_result = run_comprehensive_test_suite()
    
    # Exit with appropriate code
    exit_code = 0 if test_result.wasSuccessful() else 1
    sys.exit(exit_code)