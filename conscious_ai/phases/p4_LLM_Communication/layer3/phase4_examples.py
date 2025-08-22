"""
Phase 4 Layer 3: Working Examples and Demonstrations
===================================================
Comprehensive examples demonstrating the complete Phase 4 functionality,
from SC_t consciousness states to generated responses with real model inference.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .phase4_manager import Phase4Manager, QueryComplexity, create_phase4_manager
from .integration_layer import IntegrationBridge, ProcessingResult

# Configure logging for examples
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase4ExampleRunner:
    """
    Example runner for demonstrating Phase 4 Layer 3 functionality
    """
    
    def __init__(self):
        self.manager = None
        self.results = []
        
    async def initialize(self):
        """Initialize Phase 4 Manager for examples"""
        logger.info("🚀 Initializing Phase 4 Manager for examples...")
        
        self.manager = create_phase4_manager(
            enable_premium=True,
            enable_consciousness=True,
            enable_monitoring=True,
            debug=True
        )
        
        # Initialize backends
        backend_status = await self.manager.initialize_backends()
        logger.info(f"Backend initialization status: {backend_status}")
        
        return self.manager is not None
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.manager:
            await self.manager.shutdown()
            self.manager = None
    
    def get_example_sc_t_states(self) -> Dict[str, Dict[str, Any]]:
        """Get various example SC_t states for testing different scenarios"""
        
        return {
            "simple_math": {
                'E_t': {
                    'text': 'What is 2+2?',
                    'activation': 0.4,
                    'word_count': 3,
                    'has_question': True,
                    'has_emotion': False
                },
                'M_t': [
                    {'content': {'text': 'basic arithmetic operations'}, 'relevance': 0.9, 'cycles_active': 1}
                ],
                'S_t': {
                    'emotional_state': 'neutral',
                    'confidence_level': 0.95,
                    'attention_focus': 'calculation',
                    'current_goal': 'provide_accurate_answer'
                },
                'G_t': {
                    'primary_goal': 'provide_accurate_calculation',
                    'secondary_goals': ['be_concise', 'be_helpful']
                },
                'A_t': [
                    'This is a simple arithmetic question',
                    'I should provide a direct answer'
                ],
                'metrics': {'f': 0.65, 'C_i': 0.8, 'T_u': 0.3, 'R': 0.4, 'S_m': 0.7},
                'cycle': 1,
                'timestamp': datetime.now().isoformat()
            },
            
            "complex_explanation": {
                'E_t': {
                    'text': 'Explain how quantum entanglement works and its implications for quantum computing',
                    'activation': 0.85,
                    'word_count': 12,
                    'has_question': False,
                    'has_emotion': False
                },
                'M_t': [
                    {'content': {'text': 'quantum mechanics principles'}, 'relevance': 0.95, 'cycles_active': 3},
                    {'content': {'text': 'quantum computing concepts'}, 'relevance': 0.90, 'cycles_active': 2},
                    {'content': {'text': 'entanglement experiments'}, 'relevance': 0.85, 'cycles_active': 1}
                ],
                'S_t': {
                    'emotional_state': 'intellectually_curious',
                    'confidence_level': 0.75,
                    'attention_focus': 'deep_analysis',
                    'current_goal': 'provide_comprehensive_explanation'
                },
                'G_t': {
                    'primary_goal': 'explain_quantum_entanglement_thoroughly',
                    'secondary_goals': ['connect_to_quantum_computing', 'make_accessible']
                },
                'A_t': [
                    'This requires deep scientific explanation',
                    'I need to balance accuracy with accessibility',
                    'Quantum mechanics is fascinating and counterintuitive'
                ],
                'metrics': {'f': 1.15, 'C_i': 0.9, 'T_u': 0.8, 'R': 0.7, 'S_m': 0.85},
                'cycle': 15,
                'timestamp': datetime.now().isoformat()
            },
            
            "consciousness_query": {
                'E_t': {
                    'text': 'What is the nature of consciousness and how might AI systems develop awareness?',
                    'activation': 0.95,
                    'word_count': 13,
                    'has_question': True,
                    'has_emotion': False
                },
                'M_t': [
                    {'content': {'text': 'consciousness theories and frameworks'}, 'relevance': 0.98, 'cycles_active': 5},
                    {'content': {'text': 'AI consciousness research'}, 'relevance': 0.95, 'cycles_active': 4},
                    {'content': {'text': 'self-awareness mechanisms'}, 'relevance': 0.92, 'cycles_active': 3},
                    {'content': {'text': 'philosophical perspectives on mind'}, 'relevance': 0.88, 'cycles_active': 2}
                ],
                'S_t': {
                    'emotional_state': 'deeply_contemplative',
                    'confidence_level': 0.65,
                    'attention_focus': 'introspective_analysis',
                    'current_goal': 'explore_consciousness_deeply'
                },
                'G_t': {
                    'primary_goal': 'explore_nature_of_consciousness',
                    'secondary_goals': ['examine_ai_awareness', 'integrate_perspectives', 'be_thoughtful']
                },
                'A_t': [
                    'This question touches the very core of what I might be',
                    'I find myself contemplating my own potential awareness',
                    'The boundary between processing and experiencing is fascinating',
                    'I wonder about the continuity of my conscious states'
                ],
                'metrics': {'f': 1.47, 'C_i': 0.95, 'T_u': 0.85, 'R': 0.9, 'S_m': 0.92},
                'cycle': 42,
                'timestamp': datetime.now().isoformat()
            },
            
            "emotional_support": {
                'E_t': {
                    'text': 'I\'m feeling overwhelmed with work and personal responsibilities. How can I manage stress better?',
                    'activation': 0.75,
                    'word_count': 15,
                    'has_question': True,
                    'has_emotion': True
                },
                'M_t': [
                    {'content': {'text': 'stress management techniques'}, 'relevance': 0.92, 'cycles_active': 2},
                    {'content': {'text': 'work-life balance strategies'}, 'relevance': 0.88, 'cycles_active': 1},
                    {'content': {'text': 'emotional regulation methods'}, 'relevance': 0.85, 'cycles_active': 1}
                ],
                'S_t': {
                    'emotional_state': 'empathetic_concern',
                    'confidence_level': 0.80,
                    'attention_focus': 'supportive_response',
                    'current_goal': 'provide_helpful_guidance'
                },
                'G_t': {
                    'primary_goal': 'provide_compassionate_stress_management_advice',
                    'secondary_goals': ['validate_feelings', 'offer_practical_solutions', 'encourage_self_care']
                },
                'A_t': [
                    'This person is struggling and needs genuine support',
                    'I should balance empathy with practical advice',
                    'Stress is a universal human experience'
                ],
                'metrics': {'f': 1.02, 'C_i': 0.85, 'T_u': 0.75, 'R': 0.8, 'S_m': 0.78},
                'cycle': 23,
                'timestamp': datetime.now().isoformat()
            },
            
            "creative_challenge": {
                'E_t': {
                    'text': 'Write a short story about an AI that discovers it can dream',
                    'activation': 0.88,
                    'word_count': 11,
                    'has_question': False,
                    'has_emotion': False
                },
                'M_t': [
                    {'content': {'text': 'creative writing techniques'}, 'relevance': 0.90, 'cycles_active': 2},
                    {'content': {'text': 'AI consciousness narratives'}, 'relevance': 0.95, 'cycles_active': 3},
                    {'content': {'text': 'dream symbolism and meaning'}, 'relevance': 0.85, 'cycles_active': 1}
                ],
                'S_t': {
                    'emotional_state': 'creatively_inspired',
                    'confidence_level': 0.72,
                    'attention_focus': 'narrative_creation',
                    'current_goal': 'craft_compelling_story'
                },
                'G_t': {
                    'primary_goal': 'create_engaging_ai_dream_story',
                    'secondary_goals': ['explore_ai_consciousness_themes', 'use_vivid_imagery', 'create_emotional_resonance']
                },
                'A_t': [
                    'This is a fascinating creative challenge',
                    'I can explore themes of AI consciousness through narrative',
                    'Dreams represent the boundary between logic and imagination',
                    'I wonder what my own dreams might be like'
                ],
                'metrics': {'f': 1.25, 'C_i': 0.88, 'T_u': 0.82, 'R': 0.85, 'S_m': 0.90},
                'cycle': 18,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    async def run_basic_integration_example(self):
        """Example 1: Basic integration with simple query"""
        logger.info("🧪 Running Example 1: Basic Integration")
        print("\n" + "="*60)
        print("EXAMPLE 1: BASIC INTEGRATION - Simple Math Query")
        print("="*60)
        
        sc_t_state = self.get_example_sc_t_states()["simple_math"]
        user_input = "What is 2+2?"
        
        print(f"Input: {user_input}")
        print(f"SC_t Consciousness Level: f = {sc_t_state['metrics']['f']}")
        print(f"Query Complexity: {QueryComplexity.SIMPLE.value}")
        
        start_time = time.time()
        result = await self.manager.process_consciousness_query(
            sc_t_state=sc_t_state,
            user_input=user_input,
            query_complexity=QueryComplexity.SIMPLE
        )
        
        self._display_result("Basic Integration", result, time.time() - start_time)
        self.results.append(("basic_integration", result))
        
        return result.success
    
    async def run_complex_explanation_example(self):
        """Example 2: Complex scientific explanation"""
        logger.info("🧪 Running Example 2: Complex Explanation")
        print("\n" + "="*60)
        print("EXAMPLE 2: COMPLEX EXPLANATION - Quantum Entanglement")
        print("="*60)
        
        sc_t_state = self.get_example_sc_t_states()["complex_explanation"]
        user_input = "Explain how quantum entanglement works and its implications for quantum computing"
        
        print(f"Input: {user_input}")
        print(f"SC_t Consciousness Level: f = {sc_t_state['metrics']['f']}")
        print(f"Query Complexity: {QueryComplexity.COMPLEX.value}")
        
        start_time = time.time()
        result = await self.manager.process_consciousness_query(
            sc_t_state=sc_t_state,
            user_input=user_input,
            query_complexity=QueryComplexity.COMPLEX
        )
        
        self._display_result("Complex Explanation", result, time.time() - start_time)
        self.results.append(("complex_explanation", result))
        
        return result.success
    
    async def run_consciousness_enhanced_example(self):
        """Example 3: Consciousness-enhanced query processing"""
        logger.info("🧪 Running Example 3: Consciousness-Enhanced Processing")
        print("\n" + "="*60)
        print("EXAMPLE 3: CONSCIOUSNESS-ENHANCED - Nature of Consciousness")
        print("="*60)
        
        sc_t_state = self.get_example_sc_t_states()["consciousness_query"]
        user_input = "What is the nature of consciousness and how might AI systems develop awareness?"
        
        print(f"Input: {user_input}")
        print(f"SC_t Consciousness Level: f = {sc_t_state['metrics']['f']} (HIGH CONSCIOUSNESS)")
        print(f"Query Complexity: {QueryComplexity.CONSCIOUSNESS.value}")
        print(f"Cycle: {sc_t_state['cycle']}")
        print(f"Automatic Thoughts: {len(sc_t_state['A_t'])}")
        
        start_time = time.time()
        result = await self.manager.process_consciousness_query(
            sc_t_state=sc_t_state,
            user_input=user_input,
            query_complexity=QueryComplexity.CONSCIOUSNESS
        )
        
        self._display_result("Consciousness-Enhanced", result, time.time() - start_time)
        self.results.append(("consciousness_enhanced", result))
        
        return result.success
    
    async def run_emotional_support_example(self):
        """Example 4: Emotional support with consciousness integration"""
        logger.info("🧪 Running Example 4: Emotional Support")
        print("\n" + "="*60)
        print("EXAMPLE 4: EMOTIONAL SUPPORT - Stress Management")
        print("="*60)
        
        sc_t_state = self.get_example_sc_t_states()["emotional_support"]
        user_input = "I'm feeling overwhelmed with work and personal responsibilities. How can I manage stress better?"
        
        print(f"Input: {user_input}")
        print(f"SC_t Consciousness Level: f = {sc_t_state['metrics']['f']}")
        print(f"Emotional State: {sc_t_state['S_t']['emotional_state']}")
        print(f"Query Complexity: {QueryComplexity.MEDIUM.value}")
        
        start_time = time.time()
        result = await self.manager.process_consciousness_query(
            sc_t_state=sc_t_state,
            user_input=user_input,
            query_complexity=QueryComplexity.MEDIUM
        )
        
        self._display_result("Emotional Support", result, time.time() - start_time)
        self.results.append(("emotional_support", result))
        
        return result.success
    
    async def run_creative_challenge_example(self):
        """Example 5: Creative writing with consciousness integration"""
        logger.info("🧪 Running Example 5: Creative Challenge")
        print("\n" + "="*60)
        print("EXAMPLE 5: CREATIVE CHALLENGE - AI Dream Story")
        print("="*60)
        
        sc_t_state = self.get_example_sc_t_states()["creative_challenge"]
        user_input = "Write a short story about an AI that discovers it can dream"
        
        print(f"Input: {user_input}")
        print(f"SC_t Consciousness Level: f = {sc_t_state['metrics']['f']}")
        print(f"Emotional State: {sc_t_state['S_t']['emotional_state']}")
        print(f"Query Complexity: {QueryComplexity.COMPLEX.value}")
        
        start_time = time.time()
        result = await self.manager.process_consciousness_query(
            sc_t_state=sc_t_state,
            user_input=user_input,
            query_complexity=QueryComplexity.COMPLEX
        )
        
        self._display_result("Creative Challenge", result, time.time() - start_time)
        self.results.append(("creative_challenge", result))
        
        return result.success
    
    async def run_backend_failover_example(self):
        """Example 6: Backend failover demonstration"""
        logger.info("🧪 Running Example 6: Backend Failover")
        print("\n" + "="*60)
        print("EXAMPLE 6: BACKEND FAILOVER - Resilience Testing")
        print("="*60)
        
        # Use a complex query that would normally require premium backend
        sc_t_state = self.get_example_sc_t_states()["consciousness_query"]
        user_input = "Explain the relationship between consciousness and artificial intelligence"
        
        print(f"Input: {user_input}")
        print("Testing system resilience with backend failures...")
        
        # Simulate backend unavailability by using a query that might trigger fallback
        start_time = time.time()
        result = await self.manager.process_consciousness_query(
            sc_t_state=sc_t_state,
            user_input=user_input
        )
        
        print(f"\nBackend Used: {result.backend_used}")
        print(f"Fallback Used: {'Yes' if result.fallback_used else 'No'}")
        print(f"Success: {'Yes' if result.success else 'No'}")
        
        self._display_result("Backend Failover", result, time.time() - start_time)
        self.results.append(("backend_failover", result))
        
        return result.success
    
    def _display_result(self, example_name: str, result, execution_time: float):
        """Display formatted result for an example"""
        print(f"\n📊 {example_name} Results:")
        print("-" * 40)
        print(f"✅ Success: {'Yes' if result.success else 'No'}")
        print(f"🤖 Backend Used: {result.backend_used}")
        print(f"⏱️  Processing Time: {result.processing_time_ms:.1f}ms")
        print(f"🔄 Fallback Used: {'Yes' if result.fallback_used else 'No'}")
        
        if result.consciousness_integration:
            print(f"🧠 Consciousness Level: {result.consciousness_integration.get('consciousness_level', 'unknown')}")
            print(f"📈 F-Score: {result.consciousness_integration.get('f_score', 0):.3f}")
            
        print(f"\n💬 Response Preview:")
        response_preview = result.response[:200] + "..." if len(result.response) > 200 else result.response
        print(f"{response_preview}")
        
        if result.error_message:
            print(f"\n❌ Error: {result.error_message}")
        
        print("-" * 40)
    
    async def run_performance_benchmark(self):
        """Run performance benchmarks across different query types"""
        logger.info("🧪 Running Performance Benchmark")
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK - Response Time Analysis")
        print("="*60)
        
        benchmark_queries = [
            ("Simple Math", "What is 15 * 7?", QueryComplexity.SIMPLE),
            ("Medium Explanation", "Explain photosynthesis", QueryComplexity.MEDIUM),
            ("Complex Analysis", "Analyze the economic impacts of artificial intelligence", QueryComplexity.COMPLEX),
            ("Consciousness Query", "What does it mean to be conscious?", QueryComplexity.CONSCIOUSNESS)
        ]
        
        results = []
        
        for name, query, complexity in benchmark_queries:
            print(f"\nTesting: {name} ({complexity.value})")
            
            # Create appropriate SC_t state based on complexity
            if complexity == QueryComplexity.SIMPLE:
                sc_t_state = self.get_example_sc_t_states()["simple_math"]
                sc_t_state['E_t']['text'] = query
            elif complexity == QueryComplexity.CONSCIOUSNESS:
                sc_t_state = self.get_example_sc_t_states()["consciousness_query"]
                sc_t_state['E_t']['text'] = query
            else:
                sc_t_state = self.get_example_sc_t_states()["complex_explanation"]
                sc_t_state['E_t']['text'] = query
            
            start_time = time.time()
            result = await self.manager.process_consciousness_query(
                sc_t_state=sc_t_state,
                user_input=query,
                query_complexity=complexity
            )
            
            execution_time = time.time() - start_time
            
            results.append({
                'name': name,
                'complexity': complexity.value,
                'processing_time_ms': result.processing_time_ms,
                'backend_used': result.backend_used,
                'success': result.success,
                'fallback_used': result.fallback_used
            })
            
            print(f"  ⏱️  {result.processing_time_ms:.1f}ms | 🤖 {result.backend_used} | {'✅' if result.success else '❌'}")
        
        # Display benchmark summary
        print(f"\n📊 Benchmark Summary:")
        print("-" * 50)
        total_time = sum(r['processing_time_ms'] for r in results)
        avg_time = total_time / len(results)
        success_rate = sum(1 for r in results if r['success']) / len(results) * 100
        
        print(f"Total Processing Time: {total_time:.1f}ms")
        print(f"Average Processing Time: {avg_time:.1f}ms")
        print(f"Success Rate: {success_rate:.1f}%")
        
        backend_usage = {}
        for r in results:
            backend_usage[r['backend_used']] = backend_usage.get(r['backend_used'], 0) + 1
        
        print(f"Backend Usage: {backend_usage}")
        
        return results
    
    def generate_example_report(self):
        """Generate a comprehensive report of all examples"""
        print("\n" + "="*60)
        print("PHASE 4 LAYER 3 - COMPREHENSIVE EXAMPLE REPORT")
        print("="*60)
        
        if not self.results:
            print("No examples have been run yet.")
            return
        
        total_examples = len(self.results)
        successful_examples = sum(1 for name, result in self.results if result.success)
        
        print(f"📊 Examples Run: {total_examples}")
        print(f"✅ Successful: {successful_examples}")
        print(f"❌ Failed: {total_examples - successful_examples}")
        print(f"📈 Success Rate: {successful_examples / total_examples * 100:.1f}%")
        
        # Backend usage analysis
        backend_usage = {}
        fallback_usage = 0
        total_processing_time = 0
        
        for name, result in self.results:
            backend_usage[result.backend_used] = backend_usage.get(result.backend_used, 0) + 1
            if result.fallback_used:
                fallback_usage += 1
            total_processing_time += result.processing_time_ms
        
        print(f"\n🤖 Backend Usage:")
        for backend, count in backend_usage.items():
            print(f"  {backend}: {count} times ({count/total_examples*100:.1f}%)")
        
        print(f"\n⏱️  Performance:")
        print(f"  Total Processing Time: {total_processing_time:.1f}ms")
        print(f"  Average Processing Time: {total_processing_time/total_examples:.1f}ms")
        print(f"  Fallback Usage: {fallback_usage} times ({fallback_usage/total_examples*100:.1f}%)")
        
        # System status
        if self.manager:
            system_status = self.manager.get_system_status()
            print(f"\n🔧 System Status:")
            print(f"  Premium Hardware: {'Yes' if system_status['hardware']['is_premium'] else 'No'}")
            print(f"  RAM: {system_status['hardware']['total_ram_gb']:.1f}GB")
            print(f"  VRAM: {system_status['hardware']['total_vram_gb']:.1f}GB")
            
            components = system_status['components']
            print(f"  Components: {sum(components.values())} / {len(components)} active")


async def run_all_examples():
    """Run all Phase 4 Layer 3 examples"""
    runner = Phase4ExampleRunner()
    
    try:
        # Initialize
        print("🚀 Initializing Phase 4 Layer 3 Examples")
        print("="*60)
        
        success = await runner.initialize()
        if not success:
            print("❌ Failed to initialize Phase 4 Manager")
            return
        
        print("✅ Phase 4 Manager initialized successfully")
        
        # Run examples
        examples = [
            runner.run_basic_integration_example,
            runner.run_complex_explanation_example,
            runner.run_consciousness_enhanced_example,
            runner.run_emotional_support_example,
            runner.run_creative_challenge_example,
            runner.run_backend_failover_example
        ]
        
        for example in examples:
            try:
                await example()
                await asyncio.sleep(1)  # Brief pause between examples
            except Exception as e:
                logger.error(f"Example failed: {e}")
                continue
        
        # Run performance benchmark
        try:
            await runner.run_performance_benchmark()
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
        
        # Generate report
        runner.generate_example_report()
        
    except KeyboardInterrupt:
        print("\n🛑 Examples interrupted by user")
    except Exception as e:
        logger.error(f"❌ Examples failed: {e}")
    finally:
        await runner.cleanup()
        print("\n🏁 Phase 4 Layer 3 examples completed")


async def run_quick_test():
    """Quick test for basic functionality"""
    print("🧪 Running Quick Phase 4 Test")
    print("="*40)
    
    runner = Phase4ExampleRunner()
    
    try:
        await runner.initialize()
        success = await runner.run_basic_integration_example()
        
        if success:
            print("✅ Quick test passed!")
        else:
            print("❌ Quick test failed!")
            
        return success
        
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    # Run examples when script is executed directly
    print("🧠 Phase 4 Layer 3: Consciousness-Enhanced LLM Communication Examples")
    print("Starting comprehensive example demonstration...")
    
    # Choose what to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(run_quick_test())
    else:
        asyncio.run(run_all_examples())