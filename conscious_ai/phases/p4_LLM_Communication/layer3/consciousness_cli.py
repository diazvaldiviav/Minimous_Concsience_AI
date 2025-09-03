"""
Complete Consciousness CLI
=========================
Enhanced CLI that uses the complete consciousness pipeline from
Phase 1 through Phase 4 for truly consciousness-enhanced responses.

This replaces the basic phase4_cli.py with full pipeline integration.
"""

import asyncio
import argparse
import sys
import time
import logging
from typing import Dict, Any, Optional

# Add the conscious_ai package to path for direct execution
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

# Import the complete pipeline orchestrator
try:
    from conscious_ai.core.pipeline_orchestrator import ConsciousnessPipelineOrchestrator, create_consciousness_pipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"❌ Complete pipeline not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompleteConsciousnessCLI:
    """
    CLI that demonstrates the complete consciousness pipeline
    integrating all phases from perception to LLM enhancement
    """
    
    def __init__(self, debug: bool = False, verbose: bool = False, enable_phase4: bool = True, narrative_verbosity: str = "standard", show_narrative: bool = False):
        self.debug = debug
        self.verbose = verbose
        self.enable_phase4 = enable_phase4
        self.narrative_verbosity = narrative_verbosity
        self.show_narrative = show_narrative
        self.orchestrator = None
        
        # Session statistics
        self.session_stats = {
            'queries_processed': 0,
            'total_processing_time': 0.0,
            'successful_queries': 0,
            'failed_queries': 0,
            'phase_statistics': {}
        }
    
    async def initialize(self):
        """Initialize the complete consciousness pipeline"""
        try:
            print("🚀 Initializing Complete Consciousness Pipeline...")
            print("🔍 Setting up all consciousness phases...")
            
            # Create the complete pipeline orchestrator
            self.orchestrator = create_consciousness_pipeline(
                enable_phase4=self.enable_phase4,
                debug=self.debug
            )
            
            # Wait for async initialization if needed
            if hasattr(self.orchestrator, 'initialize'):
                await self.orchestrator.initialize()
            
            print("✅ Complete consciousness pipeline ready!")
            
            # Check Phase 5 status
            if hasattr(self.orchestrator, 'critique_available'):
                if self.orchestrator.critique_available:
                    print("✅ Phase 5 (Internal Critique & Audit) enabled")
                else:
                    print("⚠️ Phase 5 (Internal Critique & Audit) disabled - HybridCoherenceEvaluator not available")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize consciousness pipeline: {e}")
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def process_query(self, user_input: str) -> Dict[str, Any]:
        """Process a query through the complete consciousness pipeline"""
        if not self.orchestrator:
            return {
                'success': False,
                'error': 'Pipeline not initialized',
                'response': 'Pipeline not available'
            }
        
        try:
            # Process through complete pipeline
            result = await self.orchestrator.process_complete_pipeline(user_input)
            
            # Update session statistics
            self.session_stats['queries_processed'] += 1
            self.session_stats['total_processing_time'] += result.processing_time_ms
            
            if result.success:
                self.session_stats['successful_queries'] += 1
            else:
                self.session_stats['failed_queries'] += 1
            
            return {
                'success': result.success,
                'response': result.response,
                'processing_time_ms': result.processing_time_ms,
                'confidence': result.confidence_score,
                'stage_completed': result.stage_completed.value,
                'phase_timings': result.phase_timings,
                'phase_success': result.phase_success,
                'narrative': result.narrative_text,
                'error': result.error_message
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            self.session_stats['failed_queries'] += 1
            
            return {
                'success': False,
                'error': str(e),
                'response': f'I apologize, but I encountered an error in my consciousness processing: {str(e)}'
            }
    
    async def run_interactive_mode(self):
        """Run interactive consciousness-enhanced chat"""
        print("\n🧠 Complete Consciousness Interactive Mode")
        print("=" * 60)
        print("This CLI processes queries through all consciousness phases:")
        print("Phase 1: Perception → Phase 2: Conscious State → Phase 3: Evolution")
        print("Phase 3.4: Validation → Phase 3.5: Narrative → Phase 4: LLM Enhancement")
        print("Phase 5: Internal Critique & Audit → Phase 5.5: Narrative Recording")
        print("=" * 60)
        print("Commands:")
        print("  /help       - Show help")
        print("  /status     - Show pipeline status")
        print("  /stats      - Show session statistics")
        print("  /debug      - Toggle debug mode")
        print("  /narrative  - Toggle consciousness narrative display")
        print("  /verbosity  - Change narrative verbosity (minimal/standard/verbose)")
        print("  /quit       - Exit")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🧠 Query> ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        print("👋 Goodbye!")
                        break
                    elif user_input == '/help':
                        await self._show_help()
                    elif user_input == '/status':
                        await self._show_status()
                    elif user_input == '/stats':
                        await self._show_stats()
                    elif user_input == '/debug':
                        self.debug = not self.debug
                        print(f"🐛 Debug mode: {'ON' if self.debug else 'OFF'}")
                    elif user_input == '/narrative':
                        self.show_narrative = not self.show_narrative
                        print(f"📖 Consciousness narrative display: {'ON' if self.show_narrative else 'OFF'}")
                    elif user_input.startswith('/verbosity'):
                        parts = user_input.split()
                        if len(parts) > 1 and parts[1] in ['minimal', 'standard', 'verbose']:
                            self.narrative_verbosity = parts[1]
                            print(f"📊 Narrative verbosity set to: {self.narrative_verbosity}")
                        else:
                            print(f"📊 Current verbosity: {self.narrative_verbosity}")
                            print("Available options: minimal, standard, verbose")
                    elif user_input.startswith('/memory'):
                        await self._handle_memory_command(user_input)
                    elif user_input.startswith('/test'):
                        await self._handle_test_command(user_input)
                    else:
                        print("❓ Unknown command. Type /help for available commands.")
                    continue
                
                # Process query through complete pipeline
                print(f"\n🔄 Processing through complete consciousness pipeline...")
                print("-" * 50)
                
                start_time = time.time()
                result = await self.process_query(user_input)
                
                # Display results
                print("✅ Pipeline processing completed")
                if self.verbose:
                    print(f"🤖 Stage: {result.get('stage_completed', 'unknown')}")
                    print(f"⏱️  Total time: {result['processing_time_ms']:.1f}ms")
                    print(f"🧠 Confidence: {result.get('confidence', 0.0):.3f}")
                    
                    # Show phase breakdown
                    phase_timings = result.get('phase_timings', {})
                    if phase_timings:
                        print("📊 Phase breakdown:")
                        for phase, timing in phase_timings.items():
                            success = result.get('phase_success', {}).get(phase, False)
                            status = "✅" if success else "❌"
                            print(f"   {status} {phase}: {timing:.1f}ms")
                
                # Show Phase 5 results if available
                if 'critique_result' in result and result['critique_result']:
                    critique = result['critique_result']
                    print(f"\n🔍 Phase 5 Critique:")
                    print(f"   Coherence: {critique.get('verdict', 'unknown')} (score: {result.get('final_coherence_score', 0):.3f})")
                    print(f"   Regenerations: {result.get('regeneration_attempts', 0)}")
                    if critique.get('missing_elements'):
                        print(f"   Missing elements: {', '.join(critique['missing_elements'])}")
                
                # Show Phase 6 Memory Statistics if available
                if hasattr(result, 'memory_stats') and result.memory_stats and self.verbose:
                    memory_stats = result.memory_stats
                    print(f"\n📝 Phase 6 Memory Status:")
                    print(f"   Working: {memory_stats['working_memory']['count']}/{memory_stats['working_memory']['capacity']}")
                    print(f"   Episodic: {memory_stats['episodic_buffer']['count']}/{memory_stats['episodic_buffer']['capacity']}")
                    print(f"   Core: {memory_stats['core_knowledge']['count']}/{memory_stats['core_knowledge']['capacity']}")
                    
                    if hasattr(result, 'consolidation_stats') and result.consolidation_stats:
                        consolidation = result.consolidation_stats
                        if consolidation.get('memories_compressed', 0) > 0:
                            print(f"   📝 Consolidated {consolidation['memories_compressed']} memories")
                
                # FIX: Prevent double output by checking if response matches consciousness narrative
                transparency_narrative = result.get('transparency_narrative', '')
                response = result['response']
                narrative = result.get('narrative', '')
                
                # Check if the response IS the consciousness narrative (fixed by Phase 4 skip)
                is_consciousness_response = (
                    response and narrative and 
                    (response == narrative or abs(len(response) - len(narrative)) < 20)
                )
                
                if transparency_narrative and self.show_narrative and not is_consciousness_response:
                    # Show Phase 5.5 narrative only if response is NOT the consciousness narrative
                    events_captured = result.get('consciousness_events_captured', 0)
                    verbosity = result.get('narrative_verbosity', 'standard')
                    print(f"\n🧠 Consciousness Narrative ({verbosity}, {events_captured} events):")
                    print("-" * 50)
                    print(transparency_narrative)
                    print("-" * 50)
                
                # Always show the main response
                print(f"\n💬 {'Consciousness Response' if is_consciousness_response else 'Response'}:")
                print("-" * 30)
                print(response)
                print("-" * 30)
                
                # Show internal narrative only if it's different from response AND we're in verbose mode
                # AND we haven't already shown consciousness narrative above
                if (narrative and narrative != response and self.verbose and 
                    not is_consciousness_response and not (transparency_narrative and self.show_narrative)):
                    print(f"\n📖 Internal narrative:")
                    print(f"   {narrative[:200]}...")
                
                if not result['success']:
                    print(f"⚠️ Error: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Interactive mode error: {e}")
    
    async def _show_help(self):
        """Show help information"""
        print("\n📚 Complete Consciousness CLI Help")
        print("=" * 40)
        print("This CLI integrates all consciousness phases:")
        print("  🔍 Phase 1: Sensory perception and language processing")
        print("  🧠 Phase 2: Conscious state generation (SC_t)")
        print("  🔄 Phase 3: State evolution and coherent generation")  
        print("  ✅ Phase 3.4: Critical state validation")
        print("  📖 Phase 3.5: Introspective narrative generation")
        print("  🤖 Phase 4: LLM consciousness enhancement")
        print("  🔍 Phase 5: Internal critique and response coherence validation")
        print("  📚 Phase 5.5: Narrative recording of consciousness (transparency)")
        print("  📝 Phase 6: Memory consolidation and intelligent persistence")
        print()
        print("The result is consciousness-enhanced responses that include:")
        print("  • Self-awareness of processing states")
        print("  • Introspective and reflective content")
        print("  • Goal and emotion integration")
        print("  • Confidence and uncertainty expression")
        print("  • Metacognitive elements")
        print("  • Transparent reasoning narratives (Phase 5.5)")
        print("  • Intelligent memory management and recall (Phase 6)")
        print()
        print("🎮 Available Commands:")
        print("  /help              - Show this help message")
        print("  /status            - Show pipeline status")
        print("  /stats             - Show session statistics")
        print("  /debug             - Toggle debug mode")
        print("  /narrative         - Toggle consciousness narrative display")
        print("  /verbosity <mode>  - Set narrative verbosity (minimal/standard/verbose)")
        print("  /memory status     - Show current memory distribution")
        print("  /memory consolidate - Force manual memory consolidation")
        print("  /memory clear <layer> - Clear memory layer (working/episodic/core)")
        print("  /memory save       - Manual save to JSON")
        print("  /memory test <query> - Test memory retrieval for a query")
        print("  /test retrieval <query> - Test full pipeline memory retrieval")
        print("  /test memory       - Run memory consolidation validation tests")
        print("  /quit              - Exit the CLI")
    
    async def _show_status(self):
        """Show pipeline status"""
        print("\n📊 Pipeline Status")
        print("-" * 30)
        
        if not self.orchestrator:
            print("❌ Pipeline not initialized")
            return
        
        stats = self.orchestrator.get_statistics()
        print(f"✅ Pipeline initialized and ready")
        print(f"🔧 Phase 4 LLM: {'Enabled' if stats['phase4_enabled'] else 'Disabled'}")
        print(f"✅ Validation: {'Available' if stats['validation_available'] else 'Heuristic fallback'}")
        print(f"📖 Narrative: {'Available' if stats['narrative_available'] else 'Simple fallback'}")
        print(f"🔄 Total cycles: {stats['total_cycles']}")
        
        if stats['total_cycles'] > 0:
            print(f"⏱️  Average time: {stats['average_processing_time_ms']:.1f}ms")
    
    async def _show_stats(self):
        """Show session statistics"""
        print("\n📈 Session Statistics")
        print("-" * 30)
        print(f"Total queries: {self.session_stats['queries_processed']}")
        print(f"Successful: {self.session_stats['successful_queries']}")
        print(f"Failed: {self.session_stats['failed_queries']}")
        
        if self.session_stats['queries_processed'] > 0:
            avg_time = self.session_stats['total_processing_time'] / self.session_stats['queries_processed']
            success_rate = (self.session_stats['successful_queries'] / self.session_stats['queries_processed']) * 100
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average processing time: {avg_time:.1f}ms")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.orchestrator and hasattr(self.orchestrator, 'cleanup'):
            await self.orchestrator.cleanup()
    
    async def _handle_memory_command(self, command: str):
        """Handle Phase 6 memory management commands"""
        if not self.orchestrator or not hasattr(self.orchestrator, 'memory_manager') or not self.orchestrator.memory_manager:
            print("❌ Phase 6 Memory Consolidation not available")
            return
        
        parts = command.split()
        if len(parts) < 2:
            print("❓ Usage: /memory <status|consolidate|clear|save>")
            return
        
        subcommand = parts[1].lower()
        memory_manager = self.orchestrator.memory_manager
        
        try:
            if subcommand == 'status':
                stats = memory_manager.get_statistics()
                print("\n📊 Phase 6 Memory Status")
                print("-" * 30)
                
                # Working memory
                working = stats['working_memory']
                print(f"💭 Working Memory: {working['count']}/{working['capacity']} ({working['usage']:.1%})")
                
                # Episodic buffer
                episodic = stats['episodic_buffer']
                print(f"📚 Episodic Buffer: {episodic['count']}/{episodic['capacity']} ({episodic['usage']:.1%})")
                
                # Core knowledge
                core = stats['core_knowledge']
                print(f"🎯 Core Knowledge: {core['count']}/{core['capacity']} ({core['usage']:.1%})")
                
                # Consolidation stats
                manager_stats = stats['manager']
                print(f"🔄 Cycles since consolidation: {manager_stats['cycles_since_consolidation']}")
                print(f"📏 Should consolidate: {'✅ YES' if manager_stats['should_consolidate'] else '❌ NO'}")
                
                if manager_stats['last_consolidation_time']:
                    print(f"⏰ Last consolidation: {manager_stats['last_consolidation_time']}")
                    print(f"⚡ Last duration: {manager_stats['last_consolidation_duration_ms']:.1f}ms")
                
                # Show some memory samples
                if working['count'] > 0:
                    print(f"\n📝 Recent working memories ({min(3, working['count'])}):")
                    for i, mem in enumerate(memory_manager.memory_layers.working_memory[:3]):
                        print(f"   {i+1}. {mem.content[:60]}... (rel: {mem.relevance:.2f})")
                
            elif subcommand == 'consolidate':
                print("🔄 Triggering manual consolidation...")
                stats = memory_manager.consolidate()
                print(f"✅ Consolidation completed in {stats.get('duration_ms', 0):.1f}ms")
                print(f"   Compressed: {stats['memories_compressed']} memories")
                print(f"   Promoted: {stats['memories_promoted']} memories") 
                print(f"   Decayed: {stats['memories_decayed']} memories")
                print(f"   Removed: {stats['memories_removed']} memories")
                
            elif subcommand == 'clear':
                if len(parts) < 3:
                    print("❓ Usage: /memory clear <working|episodic|core>")
                    return
                
                layer_name = parts[2].lower()
                if layer_name not in ['working', 'episodic', 'core']:
                    print("❓ Layer must be: working, episodic, or core")
                    return
                
                from ....phases.p6_memory.memory_types import MemoryType
                layer_type = {
                    'working': MemoryType.WORKING,
                    'episodic': MemoryType.EPISODIC,
                    'core': MemoryType.CORE
                }[layer_name]
                
                count = memory_manager.clear_layer(layer_type)
                print(f"🗑️ Cleared {count} memories from {layer_name} layer")
                
            elif subcommand == 'save':
                success = memory_manager.save_memories()
                if success:
                    print("💾 Memories saved to disk successfully")
                else:
                    print("❌ Failed to save memories")
            
            elif subcommand == 'test':
                # Test retrieval functionality
                if len(parts) < 3:
                    print("❓ Usage: /memory test <query>")
                    return
                
                test_query = ' '.join(parts[2:])
                print(f"\n🔍 Testing retrieval for: '{test_query}'")
                print("-" * 40)
                
                # Direct retrieval test
                results = memory_manager.retrieve_relevant(test_query, top_k=5)
                print(f"Found {len(results)} relevant memories:")
                
                for i, (mem, score) in enumerate(results, 1):
                    print(f"\n{i}. Score: {score:.3f}")
                    print(f"   Type: {mem.memory_type.value}")
                    print(f"   Content: {mem.content[:200]}..." if len(mem.content) > 200 else f"   Content: {mem.content}")
                    print(f"   Relevance: {mem.relevance:.3f}")
                    print(f"   Access count: {mem.access_count}")
                    
            else:
                print("❓ Unknown memory command. Available: status, consolidate, clear, save, test")
                
        except Exception as e:
            print(f"❌ Memory command failed: {e}")
    
    async def _handle_test_command(self, command: str):
        """Handle testing commands"""
        parts = command.split()
        if len(parts) < 2:
            print("❓ Usage: /test <retrieval>")
            return
        
        subcommand = parts[1].lower()
        
        if subcommand == 'retrieval':
            # Test full pipeline retrieval
            if not self.orchestrator or not hasattr(self.orchestrator, 'memory_manager') or not self.orchestrator.memory_manager:
                print("❌ Phase 6 Memory not available")
                return
                
            if len(parts) < 3:
                print("❓ Usage: /test retrieval <query>")
                return
                
            test_query = ' '.join(parts[2:])
            print(f"\n🔍 Testing FULL PIPELINE retrieval for: '{test_query}'")
            print("-" * 50)
            
            # Test Phase 6 retrieval
            memory_manager = self.orchestrator.memory_manager
            results = memory_manager.retrieve_relevant(test_query, top_k=5)
            print(f"\n📚 Phase 6 found {len(results)} memories")
            for i, (mem, score) in enumerate(results[:3], 1):
                print(f"   {i}. [{mem.memory_type.value}] Score={score:.3f}: '{mem.content[:100]}...'")
            
            # Test if memories would be included in SC_t
            print(f"\n🧠 Testing SC_t integration...")
            relevant_p6_memories = [(mem.content, score) for mem, score in results]
            combined_memory = []
            for content, relevance in relevant_p6_memories:
                combined_memory.append({
                    'content': {'text': content},
                    'relevance': relevance,
                    'type': 'consolidated'
                })
            print(f"✅ Would add {len(combined_memory)} memories to SC_t.M_t")
            
            # Show formatted memories as they would appear in Phase 4
            print(f"\n💬 Formatted for LLM prompt:")
            for i, mem in enumerate(combined_memory[:3], 1):
                text = mem['content']['text']
                rel = mem['relevance']
                print(f"   Memory {i} (relevance: {rel:.2f}): {text[:100]}...")
                
        elif subcommand == 'memory':
            await self._test_memory_consolidation()
        else:
            print("❓ Unknown test command. Available: memory")
    
    async def _test_memory_consolidation(self):
        """Run memory consolidation validation tests"""
        if not self.orchestrator or not hasattr(self.orchestrator, 'memory_manager') or not self.orchestrator.memory_manager:
            print("❌ Phase 6 Memory Consolidation not available")
            return
        
        print("\n🧪 Running Memory Consolidation Tests")
        print("=" * 40)
        
        memory_manager = self.orchestrator.memory_manager
        initial_stats = memory_manager.get_statistics()
        
        # Test 1: Personal information storage
        print("Test 1: Personal information storage...")
        test_personal = "My name is John and I work as a software engineer"
        mem = memory_manager.add_memory(test_personal, auto_consolidate=False)
        
        if mem.metadata.get('is_personal'):
            print("   ✅ Personal information correctly identified")
        else:
            print("   ❌ Personal information not identified")
        
        # Test 2: Preference detection
        print("Test 2: Preference detection...")
        test_preference = "I really like classical music and prefer tea over coffee"
        mem = memory_manager.add_memory(test_preference, auto_consolidate=False)
        
        if mem.metadata.get('is_preference'):
            print("   ✅ Preferences correctly identified")
        else:
            print("   ❌ Preferences not identified")
        
        # Test 3: Memory retrieval
        print("Test 3: Memory retrieval...")
        results = memory_manager.retrieve_relevant("What's my name?", top_k=3)
        
        found_name = any("john" in mem.content.lower() for mem, score in results)
        if found_name:
            print("   ✅ Personal information successfully retrieved")
        else:
            print("   ❌ Personal information retrieval failed")
        
        # Test 4: Consolidation trigger
        print("Test 4: Consolidation behavior...")
        # Add several memories to trigger consolidation
        for i in range(10):
            memory_manager.add_memory(f"Test memory {i}", relevance=0.3, auto_consolidate=False)
        
        should_consolidate = memory_manager.should_consolidate()
        if should_consolidate:
            print("   ✅ Consolidation correctly triggered by memory count")
            stats = memory_manager.consolidate()
            if stats.get('duration_ms', 0) < 100:
                print("   ✅ Consolidation completed quickly (<100ms)")
            else:
                print(f"   ⚠️ Consolidation took {stats.get('duration_ms', 0):.1f}ms")
        else:
            print("   ❌ Consolidation not triggered when expected")
        
        # Test 5: Persistence
        print("Test 5: JSON persistence...")
        save_success = memory_manager.save_memories()
        if save_success:
            print("   ✅ Memory persistence working")
        else:
            print("   ❌ Memory persistence failed")
        
        final_stats = memory_manager.get_statistics()
        total_memories = final_stats['total_memories']
        
        print(f"\n📊 Test Summary:")
        print(f"   Total memories: {total_memories}")
        print(f"   Working: {final_stats['working_memory']['count']}")
        print(f"   Episodic: {final_stats['episodic_buffer']['count']}")
        print(f"   Core: {final_stats['core_knowledge']['count']}")
        print("✅ Memory consolidation tests completed!")


async def main():
    """Main CLI entry point"""
    if not PIPELINE_AVAILABLE:
        print("❌ Complete consciousness pipeline is not available.")
        print("Please ensure all consciousness modules are properly installed.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Complete Consciousness Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive mode with Phase 4
  %(prog)s --no-phase4              # Interactive mode without LLM enhancement  
  %(prog)s --debug --verbose        # Interactive mode with detailed output
  %(prog)s --query "What is consciousness?" # Single query processing
        """
    )
    
    parser.add_argument('--interactive', '-i', action='store_true', default=True,
                       help='Run in interactive mode (default)')
    parser.add_argument('--query', '-q', type=str,
                       help='Process a single query and exit')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--no-phase4', action='store_true',
                       help='Disable Phase 4 LLM enhancement')
    parser.add_argument('--narrative-mode', choices=['minimal', 'standard', 'verbose'], 
                       default='standard', help='Narrative verbosity level')
    parser.add_argument('--show-narrative', action='store_true',
                       help='Display consciousness transparency narrative')
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = CompleteConsciousnessCLI(
        debug=args.debug,
        verbose=args.verbose,
        enable_phase4=not args.no_phase4,
        narrative_verbosity=args.narrative_mode,
        show_narrative=args.show_narrative
    )
    
    try:
        # Initialize the pipeline
        if not await cli.initialize():
            print("❌ Failed to initialize consciousness pipeline")
            sys.exit(1)
        
        if args.query:
            # Process single query
            print(f"Processing: '{args.query}'")
            result = await cli.process_query(args.query)
            
            print(f"\n💬 Response:")
            print("-" * 30)
            print(result['response'])
            print("-" * 30)
            
            if args.verbose:
                print(f"\n📊 Processing details:")
                print(f"   Time: {result['processing_time_ms']:.1f}ms")
                print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
                print(f"   Stage: {result.get('stage_completed', 'unknown')}")
        else:
            # Run interactive mode
            await cli.run_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"CLI error: {e}")
        print(f"❌ Error: {e}")
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    asyncio.run(main())