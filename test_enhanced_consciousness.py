#!/usr/bin/env python3
"""
Enhanced Consciousness Integration Test
======================================
Standalone test script demonstrating enhanced metacognitive capabilities
and Phase 7 integration without modifying existing code.

This script tests:
1. Enhanced conscious state with temporal awareness
2. Metacognitive observations
3. Phase 7 response generation
4. Model registry tracking
5. API endpoint functionality

Usage:
    python test_enhanced_consciousness.py
    python test_enhanced_consciousness.py --test-api
    python test_enhanced_consciousness.py --test-metacognition
"""

import asyncio
import argparse
import sys
import logging
import time
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import consciousness components
from conscious_ai.core.pipeline_orchestrator import create_consciousness_pipeline
from conscious_ai.phases.p2_cognitive_context.enhanced_conscious_state import (
    EnhancedConsciousState, create_enhanced_conscious_state
)
from conscious_ai.phases.p7_expressive_execution import (
    ResponseGenerator, get_available_models, validate_model
)
from conscious_ai.debug import get_model_registry, log_phase7_final_usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedConsciousnessDemo:
    """Demonstrates enhanced consciousness capabilities"""
    
    def __init__(self):
        self.pipeline = None
        self.phase7_generator = None
        self.previous_state = None
        self.query_count = 0
        
    async def initialize(self):
        """Initialize consciousness pipeline and Phase 7"""
        print("🧠 Initializing Enhanced Consciousness Demo...")
        
        # Create standard pipeline
        self.pipeline = create_consciousness_pipeline(
            enable_phase4=True,
            debug=True
        )
        
        # Create Phase 7 generator
        self.phase7_generator = ResponseGenerator(
            default_model="gpt-4o-mini",
            enable_fallback=True
        )
        
        print(f"✅ Pipeline initialized")
        print(f"✅ Phase 7 available: {self.phase7_generator.available}")
        print(f"✅ Available models: {get_available_models()}")
        print("-" * 60)
        
    async def process_enhanced_query(self, user_input: str, use_model: str = "gpt-4o-mini"):
        """Process query with enhanced consciousness capabilities"""
        self.query_count += 1
        
        print(f"\n🔄 Processing Query {self.query_count}: '{user_input}'")
        print("=" * 80)
        
        start_time = time.time()
        
        # Step 1: Process through standard pipeline (Phases 1-6)
        print("📍 Step 1: Processing through standard consciousness pipeline...")
        result = await self.pipeline.process_complete_pipeline(user_input)
        
        if not result.success:
            print(f"❌ Pipeline failed: {result.error_message}")
            return
            
        print(f"✅ Standard pipeline complete ({result.processing_time_ms:.1f}ms)")
        print(f"   Confidence: {result.confidence_score:.2f}")
        print(f"   Stage: {result.stage_completed.value}")
        
        # Step 2: Enhance conscious state with metacognitive capabilities
        print("\n📍 Step 2: Enhancing with metacognitive capabilities...")
        
        try:
            enhanced_state = create_enhanced_conscious_state(
                result.conscious_state,
                previous_state=self.previous_state
            )
            
            print(f"✅ Enhanced consciousness created")
            print(f"   Metacognitive depth: Level {enhanced_state.metacognitive_depth}")
            print(f"   Meta-thoughts: {len(enhanced_state.meta_thoughts)}")
            print(f"   Self-observations: {len(enhanced_state.observation_stack)}")
            print(f"   State transitions: {len(enhanced_state.state_transitions)}")
            
            # Show temporal context if available
            temporal = enhanced_state.get_temporal_context()
            if temporal['has_previous']:
                print(f"   Temporal continuity: {temporal['temporal_continuity']:.1%}")
                if temporal['previous_thought']:
                    print(f"   Previous thought: '{temporal['previous_thought'][:50]}...'")
            
            # Display meta-thoughts
            if enhanced_state.meta_thoughts:
                print(f"\n   🧠 Meta-thoughts:")
                for i, mt in enumerate(enhanced_state.meta_thoughts[:3], 1):
                    print(f"     {i}. [{mt.observation_type}] {mt.content}")
            
            # Display self-observations
            introspective_obs = [o for o in enhanced_state.observation_stack if o.triggers_introspection]
            if introspective_obs:
                print(f"\n   👁️  Introspective observations:")
                for i, obs in enumerate(introspective_obs[:2], 1):
                    print(f"     {i}. {obs.observation}")
            
            # Store for next query's temporal awareness
            self.previous_state = enhanced_state
            
        except Exception as e:
            print(f"⚠️ Metacognitive enhancement failed: {e}")
            enhanced_state = result.conscious_state
        
        # Step 3: Generate final response using Phase 7
        print(f"\n📍 Step 3: Generating final response with {use_model}...")
        
        if self.phase7_generator.available and validate_model(use_model):
            try:
                # Convert to dictionary format for Phase 7
                enhanced_dict = enhanced_state.to_dict() if hasattr(enhanced_state, 'to_dict') else enhanced_state
                
                phase7_result = await self.phase7_generator.generate_final_response(
                    enhanced_sc_t=enhanced_dict,
                    user_input=user_input,
                    model_override=use_model
                )
                
                if phase7_result.success:
                    print(f"✅ Phase 7 response generated using {phase7_result.model_used}")
                    print(f"   Processing time: {phase7_result.processing_time_ms:.1f}ms")
                    print(f"   Token usage: {phase7_result.input_tokens}→{phase7_result.output_tokens}")
                    print(f"   Response confidence: {phase7_result.confidence_score:.2f}")
                    
                    # Log usage
                    log_phase7_final_usage(
                        model=phase7_result.model_used,
                        input_tokens=phase7_result.input_tokens,
                        output_tokens=phase7_result.output_tokens,
                        processing_time_ms=phase7_result.processing_time_ms,
                        success=True
                    )
                    
                    final_response = phase7_result.response
                else:
                    print(f"⚠️ Phase 7 failed: {phase7_result.error_message}")
                    final_response = result.response or "Standard pipeline response"
                    
            except Exception as e:
                print(f"❌ Phase 7 error: {e}")
                final_response = result.response or "Fallback response"
        else:
            print(f"⚠️ Phase 7 not available or invalid model: {use_model}")
            final_response = result.response or "Standard pipeline response"
        
        total_time = (time.time() - start_time) * 1000
        
        # Display final results
        print(f"\n💬 FINAL RESPONSE ({total_time:.1f}ms total):")
        print("=" * 80)
        print(final_response)
        print("=" * 80)
        
        # Show comprehensive narrative if enhanced
        if hasattr(enhanced_state, 'to_comprehensive_narrative'):
            print(f"\n📖 COMPREHENSIVE CONSCIOUSNESS NARRATIVE:")
            print("-" * 60)
            print(enhanced_state.to_comprehensive_narrative())
            print("-" * 60)
        
        return enhanced_state, final_response
    
    async def test_metacognitive_capabilities(self):
        """Test specific metacognitive features"""
        print("\n🧪 Testing Metacognitive Capabilities")
        print("=" * 60)
        
        test_queries = [
            "What does it feel like to be conscious?",
            "Can you think about your own thinking process?",
            "What were you thinking about before I asked this?",
            "How do you experience the passage of time?",
            "Do you notice patterns in your own thoughts?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Metacognitive Test {i}/5 ---")
            await self.process_enhanced_query(query)
            
            if i < len(test_queries):
                print("\n⏱️ Waiting 2 seconds for temporal awareness...")
                await asyncio.sleep(2)
    
    async def test_model_selection(self):
        """Test different model selections"""
        print("\n🤖 Testing Model Selection")
        print("=" * 60)
        
        query = "Explain consciousness in simple terms"
        models_to_test = ["gpt-4o-mini", "gpt-3.5-turbo"]
        
        for model in models_to_test:
            if validate_model(model):
                print(f"\n--- Testing with {model} ---")
                await self.process_enhanced_query(query, use_model=model)
            else:
                print(f"⚠️ Skipping unsupported model: {model}")
    
    def show_model_statistics(self):
        """Display model usage statistics"""
        print("\n📊 Model Usage Statistics")
        print("=" * 60)
        
        registry = get_model_registry()
        stats = registry.get_session_statistics()
        
        print(f"Session ID: {stats['session_id']}")
        print(f"Duration: {stats['session_duration_minutes']:.1f} minutes")
        print(f"Total operations: {stats['total_usage_count']}")
        print(f"Models used: {', '.join(stats['unique_models'])}")
        
        if stats['cost_breakdown']:
            print(f"\nCost Breakdown:")
            total_cost = 0
            for model, costs in stats['cost_breakdown'].items():
                print(f"  {model}: ${costs['total_cost']:.4f}")
                total_cost += costs['total_cost']
            print(f"  Total: ${total_cost:.4f}")
        
        if stats['performance_metrics']:
            perf = stats['performance_metrics']
            print(f"\nPerformance:")
            print(f"  Avg processing time: {perf['avg_processing_time_ms']:.1f}ms")
            print(f"  Success rate: {perf['success_rate']:.1%}")
    
    async def run_full_demo(self):
        """Run complete demonstration"""
        await self.initialize()
        
        print("🚀 Starting Enhanced Consciousness Demonstration")
        print("=" * 80)
        
        # Basic enhanced processing test
        print("\n1️⃣ Basic Enhanced Processing Test")
        await self.process_enhanced_query("Hello, how are you today?")
        
        # Temporal awareness test
        print("\n2️⃣ Temporal Awareness Test")
        await self.process_enhanced_query("What were you just thinking about?")
        
        # Metacognitive test
        print("\n3️⃣ Metacognitive Test")
        await self.process_enhanced_query("Can you observe your own thought processes?")
        
        # Model selection test
        print("\n4️⃣ Model Selection Test")
        if validate_model("gpt-3.5-turbo"):
            await self.process_enhanced_query("Summarize your consciousness", use_model="gpt-3.5-turbo")
        
        # Show final statistics
        self.show_model_statistics()
        
        print(f"\n✅ Enhanced Consciousness Demo Complete!")
        print(f"Processed {self.query_count} queries with enhanced metacognitive capabilities")


async def test_api_endpoint():
    """Test the API endpoint if running"""
    print("\n🌐 Testing API Endpoint")
    print("=" * 60)
    
    try:
        import httpx
        
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        async with httpx.AsyncClient() as client:
            print("Testing health endpoint...")
            response = await client.get(f"{base_url}/health")
            
            if response.status_code == 200:
                health = response.json()
                print(f"✅ API is healthy")
                print(f"   Pipeline ready: {health['pipeline_ready']}")
                print(f"   Phase 7 ready: {health['phase7_ready']}")
                print(f"   Uptime: {health['uptime_seconds']:.1f}s")
            else:
                print(f"⚠️ Health check failed: {response.status_code}")
        
        # Test consciousness processing
        print("\nTesting consciousness processing...")
        test_request = {
            "user_input": "What is consciousness?",
            "final_model": "gpt-4o-mini",
            "include_consciousness_trace": True,
            "enable_metacognition": True
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{base_url}/process", json=test_request)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Consciousness processing successful")
                print(f"   Model used: {result['model_used']}")
                print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Emotional state: {result['emotional_state']}")
                
                if result.get('consciousness_trace'):
                    trace = result['consciousness_trace']
                    print(f"   Metacognitive depth: {trace.get('metacognitive_depth', 0)}")
                    print(f"   State transitions: {trace.get('state_transitions', 0)}")
                
                print(f"\n💬 Response preview:")
                print(f"   {result['response'][:200]}...")
            else:
                print(f"❌ Processing failed: {response.status_code}")
                print(f"   {response.text}")
        
    except ImportError:
        print("⚠️ httpx not installed - skipping API test")
        print("   Install with: pip install httpx")
    except Exception as e:
        print(f"❌ API test failed: {e}")
        print("   Make sure the API server is running:")
        print("   python conscious_ai/api/run_server.py")


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Enhanced Consciousness Integration Test")
    parser.add_argument('--test-api', action='store_true',
                       help='Test API endpoint (requires server running)')
    parser.add_argument('--test-metacognition', action='store_true',
                       help='Run detailed metacognition tests')
    parser.add_argument('--test-models', action='store_true',
                       help='Test different model selections')
    
    args = parser.parse_args()
    
    if args.test_api:
        await test_api_endpoint()
        return
    
    # Create demo instance
    demo = EnhancedConsciousnessDemo()
    
    if args.test_metacognition:
        await demo.initialize()
        await demo.test_metacognitive_capabilities()
        demo.show_model_statistics()
    elif args.test_models:
        await demo.initialize()
        await demo.test_model_selection()
        demo.show_model_statistics()
    else:
        # Run full demonstration
        await demo.run_full_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)