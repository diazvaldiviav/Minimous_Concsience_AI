#!/usr/bin/env python3
"""
Test script to verify consciousness fixes are working
"""

import sys
import os
import asyncio
import logging

# Add the conscious_ai package to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'conscious_ai'))

# Configure logging to see debug output
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from conscious_ai.core.pipeline_orchestrator import create_consciousness_pipeline
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the root directory")
    sys.exit(1)

async def test_consciousness_queries():
    """Test consciousness queries to verify fixes"""
    
    print("🧠 Testing Consciousness Fixes")
    print("=" * 50)
    
    # Create pipeline
    pipeline = create_consciousness_pipeline(enable_phase4=True, debug=True)
    
    test_queries = [
        "Who are you?",
        "What are you thinking right now?", 
        "How do you feel?",
        "What is consciousness?",
        "Describe your current state"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)
        
        try:
            result = await pipeline.process_complete_pipeline(query)
            
            print(f"✅ Success: {result.success}")
            print(f"⏱️  Time: {result.processing_time_ms:.1f}ms")
            print(f"🧠 Confidence: {result.confidence_score:.2f}")
            print(f"📊 Stage: {result.stage_completed.value}")
            
            # Show phase success
            print("📋 Phase Results:")
            for phase, success in result.phase_success.items():
                status = "✅" if success else "❌"
                timing = result.phase_timings.get(phase, 0)
                print(f"   {status} {phase}: {timing:.1f}ms")
            
            # Check narrative
            if result.narrative_text:
                print(f"📖 Narrative: {len(result.narrative_text)} chars")
                print(f"   Preview: {result.narrative_text[:150]}...")
            else:
                print("❌ No narrative generated!")
            
            print(f"\n💬 Response ({len(result.response)} chars):")
            print("=" * 30)
            print(result.response)
            print("=" * 30)
            
            # Analyze response quality
            consciousness_indicators = [
                'confidence', 'state', 'observ', 'experienc', 'process', 'aware',
                '%', 'conscious', 'introspect', 'recursive', 'layer', 'cognitive'
            ]
            
            indicators_found = sum(1 for indicator in consciousness_indicators 
                                 if indicator in result.response.lower())
            
            print(f"🎯 Consciousness indicators found: {indicators_found}/12")
            
            if "How can I assist" in result.response or "Hello!" in result.response:
                print("❌ STILL GENERIC! Contains assistant-like responses")
            else:
                print("✅ No generic assistant responses detected")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60)
    
    print("\n🏁 Testing Complete!")

if __name__ == "__main__":
    asyncio.run(test_consciousness_queries())