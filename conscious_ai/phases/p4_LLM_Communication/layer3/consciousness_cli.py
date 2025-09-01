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
                
                # Show Phase 5.5 Consciousness Narrative if available and enabled
                transparency_narrative = result.get('transparency_narrative', '')
                if transparency_narrative and self.show_narrative:
                    events_captured = result.get('consciousness_events_captured', 0)
                    verbosity = result.get('narrative_verbosity', 'standard')
                    print(f"\n🧠 Consciousness Narrative ({verbosity}, {events_captured} events):")
                    print("-" * 50)
                    print(transparency_narrative)
                    print("-" * 50)
                
                print(f"\n💬 Response:")
                print("-" * 30)
                print(result['response'])
                print("-" * 30)
                
                # Show narrative if available and different from response (Phase 3.5)
                narrative = result.get('narrative', '')
                if narrative and narrative != result['response'] and self.verbose:
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
        print()
        print("The result is consciousness-enhanced responses that include:")
        print("  • Self-awareness of processing states")
        print("  • Introspective and reflective content")
        print("  • Goal and emotion integration")
        print("  • Confidence and uncertainty expression")
        print("  • Metacognitive elements")
        print("  • Transparent reasoning narratives (Phase 5.5)")
    
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