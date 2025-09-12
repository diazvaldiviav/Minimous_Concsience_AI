#!/usr/bin/env python3
"""
Manual test script for Week 2 conversation consolidation pipeline.

This script tests the complete consolidation workflow:
MEP proposal → truth validation → training data → LoRA adapter
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.mep.schemas import MEPProposalRequest
from memory.consolidator import create_memory_consolidator
from core.config import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_proposal() -> MEPProposalRequest:
    """
    Create a sample MEP proposal for testing.
    
    Returns:
        Sample MEP proposal with realistic conversation data
    """
    return MEPProposalRequest(
        provider="anthropic",
        model="claude-3-sonnet",
        external_user_id="test_user_123",
        external_chat_id="test_chat_456",
        event_id="test_event_789",
        trigger="context_full",
        context_fill=0.85,
        token_usage={
            "window_tokens": 8000,
            "used_tokens": 6800,
            "max_tokens": 10000
        },
        message_span={
            "from_turn": 0,
            "to_turn": 15
        },
        summary_text=(
            "User discussed building a web application with React frontend and Python backend. "
            "They expressed preference for FastAPI over Django for the API. Database requirements "
            "include PostgreSQL for main data and Redis for caching. User wants to implement "
            "authentication using JWT tokens. They also mentioned needing Docker containerization "
            "for deployment and preference for dark mode in the UI."
        ),
        key_facts=[
            {
                "claim": "User prefers React for frontend development",
                "importance": 0.9,
                "confidence": 0.95,
                "category": "preference"
            },
            {
                "claim": "User wants FastAPI instead of Django for backend API",
                "importance": 0.8,
                "confidence": 0.9,
                "category": "preference"
            },
            {
                "claim": "Database should use PostgreSQL for main data",
                "importance": 0.9,
                "confidence": 0.95,
                "category": "requirement"
            },
            {
                "claim": "Redis needed for caching functionality",
                "importance": 0.7,
                "confidence": 0.8,
                "category": "requirement"
            },
            {
                "claim": "Authentication should use JWT tokens",
                "importance": 0.8,
                "confidence": 0.85,
                "category": "requirement"
            },
            {
                "claim": "User prefers dark mode for UI",
                "importance": 0.6,
                "confidence": 0.7,
                "category": "preference"
            }
        ]
    )


async def test_conversation_consolidation():
    """
    Test the complete consolidation pipeline.
    """
    print("="*60)
    print("SC Memory System - Week 2 Consolidation Test")
    print("="*60)
    
    try:
        # 1. Create sample proposal
        print("\n1. Creating sample MEP proposal...")
        proposal = create_sample_proposal()
        print(f"   ✓ Created proposal for conversation {proposal.external_chat_id}")
        print(f"   ✓ Summary length: {len(proposal.summary_text)} chars")
        print(f"   ✓ Key facts: {len(proposal.key_facts)}")
        
        # 2. Initialize consolidator
        print("\n2. Initializing memory consolidator...")
        consolidator = create_memory_consolidator()
        print("   ✓ Consolidator initialized")
        
        # 3. Run consolidation
        print("\n3. Starting conversation consolidation...")
        print("   (This may take several minutes for model loading and training)")
        
        start_time = datetime.now()
        result = await consolidator.consolidate_conversation(proposal)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # 4. Display results
        print(f"\n4. Consolidation Results:")
        print(f"   Status: {result.status}")
        
        if result.status == "success":
            print(f"   ✓ Adapter ID: {result.adapter_id}")
            print(f"   ✓ Validated facts: {result.validated_facts_count}")
            print(f"   ✓ Training examples: {result.training_examples_count}")
            print(f"   ✓ Training time: {result.training_time_seconds:.2f}s")
            print(f"   ✓ Total processing time: {processing_time:.2f}s")
            
            # 5. Test adapter recall (basic check)
            print(f"\n5. Testing adapter recall...")
            print("   Note: Full recall testing requires model inference setup")
            print("   ✓ Adapter files should be saved in ./models/adapters/")
            
            # Check if adapter files exist
            adapters_dir = Path("./models/adapters")
            if adapters_dir.exists():
                adapter_dirs = [d for d in adapters_dir.iterdir() if d.is_dir()]
                print(f"   ✓ Found {len(adapter_dirs)} adapter directories")
                
                if adapter_dirs:
                    latest_adapter = max(adapter_dirs, key=lambda p: p.stat().st_ctime)
                    print(f"   ✓ Latest adapter: {latest_adapter.name}")
                    
                    # Check adapter files
                    adapter_files = list(latest_adapter.glob("*"))
                    print(f"   ✓ Adapter files: {len(adapter_files)}")
                    for file in adapter_files:
                        print(f"     - {file.name}")
            
        else:
            print(f"   ✗ Consolidation failed: {result.error_message}")
        
        # 6. Display consolidator stats
        print(f"\n6. Consolidator Statistics:")
        stats = consolidator.get_consolidation_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n{'='*60}")
        if result.status == "success":
            print("✅ Week 2 Consolidation Test: SUCCESS")
            print("\nThe SC Memory System successfully:")
            print("- Validated conversation facts")
            print("- Generated training data from conversation")
            print("- Trained LoRA adapter on conversation data")
            print("- Saved adapter for future memory recall")
        else:
            print("❌ Week 2 Consolidation Test: FAILED")
            print(f"Error: {result.error_message}")
        
        print(f"{'='*60}")
        
        return result.status == "success"
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        logger.error("Consolidation test failed", exc_info=True)
        return False


async def main():
    """Main test function."""
    print("Starting Week 2 consolidation test...")
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the sc-memory-system directory")
        return
    
    success = await test_conversation_consolidation()
    
    if success:
        print("\n🎉 All tests passed! Week 2 implementation is working.")
    else:
        print("\n⚠️  Some tests failed. Check logs for details.")


if __name__ == "__main__":
    asyncio.run(main())