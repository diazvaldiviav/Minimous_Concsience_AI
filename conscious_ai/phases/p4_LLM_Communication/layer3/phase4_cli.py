#!/usr/bin/env python3
"""
Phase 4 Layer 3: Command-Line Interface
=======================================
Interactive CLI tool for testing and demonstrating Phase 4 functionality.
Provides real-time testing of consciousness → response pipeline with
performance monitoring and debug capabilities.
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import traceback

# Using relative imports - no sys.path manipulation needed

from .phase4_manager import Phase4Manager, QueryComplexity, create_phase4_manager
from .phase4_examples import Phase4ExampleRunner

# Configure CLI logging
logging.basicConfig(
    level=logging.WARNING,  # Keep quiet by default
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class Phase4CLI:
    """
    Command-Line Interface for Phase 4 Layer 3 testing and demonstration
    """
    
    def __init__(self, debug: bool = False, verbose: bool = False, selected_model: str = 'auto'):
        self.debug = debug
        self.verbose = verbose
        self.selected_model = selected_model
        self.manager = None
        self.session_stats = {
            'queries_processed': 0,
            'successful_queries': 0,
            'total_processing_time_ms': 0.0,
            'session_start': time.time()
        }
        
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    async def initialize(self) -> bool:
        """Initialize Phase 4 Manager"""
        try:
            print("🚀 Initializing Phase 4 Layer 3 CLI...")
            
            self.manager = create_phase4_manager(
                enable_premium=True,
                enable_consciousness=True,
                enable_monitoring=True,
                debug=self.debug,
                selected_model=self.selected_model
            )
            
            print("🔍 Detecting hardware configuration...")
            system_status = self.manager.get_system_status()
            
            if system_status['hardware']['is_premium']:
                print(f"✅ Premium hardware detected: {system_status['hardware']['total_ram_gb']:.1f}GB RAM + {system_status['hardware']['total_vram_gb']:.1f}GB VRAM")
            else:
                print("ℹ️ Standard hardware detected - limited backend options")
            
            print("🤖 Initializing backends...")
            backend_status = await self.manager.initialize_backends()
            
            available_backends = [k.value for k, v in backend_status.items() if v]
            if available_backends:
                print(f"✅ Available backends: {', '.join(available_backends)}")
            else:
                print("⚠️ No backends available - using fallback mode")
            
            print("✅ Phase 4 CLI ready!\n")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            if self.debug:
                traceback.print_exc()
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.manager:
            await self.manager.shutdown()
            self.manager = None
    
    def create_interactive_sc_t(self, user_input: str) -> Dict[str, Any]:
        """Create an interactive SC_t state based on user input"""
        
        # Analyze input for complexity and emotional content
        word_count = len(user_input.split())
        has_question = '?' in user_input
        has_emotion = any(word in user_input.lower() for word in [
            'feel', 'emotion', 'sad', 'happy', 'angry', 'frustrated', 'excited',
            'worried', 'anxious', 'stressed', 'overwhelmed', 'confused'
        ])
        
        # Determine consciousness level based on input characteristics
        consciousness_keywords = [
            'consciousness', 'awareness', 'experience', 'introspection', 'self-reflection',
            'conciencia', 'experiencia', 'reflexión', 'introspección', 'think', 'feel',
            'understand', 'realize', 'contemplate', 'ponder', 'wonder'
        ]
        
        consciousness_indicators = sum(1 for word in consciousness_keywords if word in user_input.lower())
        
        # Calculate activation and f-score based on input complexity
        activation = min(0.95, 0.3 + (word_count * 0.05) + (consciousness_indicators * 0.1))
        f_score = min(1.5, 0.5 + (consciousness_indicators * 0.2) + (word_count * 0.02))
        
        # Determine emotional state
        if has_emotion:
            if any(word in user_input.lower() for word in ['sad', 'worried', 'anxious', 'stressed', 'overwhelmed']):
                emotional_state = 'concerned'
            elif any(word in user_input.lower() for word in ['happy', 'excited']):
                emotional_state = 'positive'
            else:
                emotional_state = 'emotionally_engaged'
        elif consciousness_indicators > 2:
            emotional_state = 'contemplative'
        else:
            emotional_state = 'neutral'
        
        # Generate automatic thoughts
        automatic_thoughts = []
        if consciousness_indicators > 0:
            automatic_thoughts.append("This question makes me reflect on my own processing")
        if has_question:
            automatic_thoughts.append("I should provide a thoughtful and helpful response")
        if word_count > 15:
            automatic_thoughts.append("This is a complex query requiring careful consideration")
        if not automatic_thoughts:
            automatic_thoughts.append("Processing this request with focus and attention")
        
        return {
            'E_t': {
                'text': user_input,
                'activation': activation,
                'word_count': word_count,
                'has_question': has_question,
                'has_emotion': has_emotion
            },
            'M_t': [
                {'content': {'text': 'relevant knowledge and context'}, 'relevance': 0.85, 'cycles_active': 1},
                {'content': {'text': 'previous conversation patterns'}, 'relevance': 0.70, 'cycles_active': 2}
            ],
            'S_t': {
                'emotional_state': emotional_state,
                'confidence_level': min(0.95, 0.6 + (f_score * 0.2)),
                'attention_focus': 'user_query_analysis',
                'current_goal': 'provide_helpful_response'
            },
            'G_t': {
                'primary_goal': 'understand_and_respond_helpfully',
                'secondary_goals': ['be_accurate', 'be_thoughtful', 'be_engaging']
            },
            'A_t': automatic_thoughts,
            'metrics': {
                'f': f_score,
                'C_i': min(1.0, 0.7 + (consciousness_indicators * 0.1)),
                'T_u': 0.6,
                'R': 0.5,
                'S_m': min(1.0, 0.6 + (consciousness_indicators * 0.15))
            },
            'cycle': self.session_stats['queries_processed'] + 1,
            'timestamp': datetime.now().isoformat()
        }
    
    async def process_query(self, user_input: str, complexity: Optional[str] = None) -> Dict[str, Any]:
        """Process a single query through Phase 4"""
        
        print(f"\n🔄 Processing: '{user_input}'")
        print("-" * 50)
        
        # Create SC_t state
        sc_t_state = self.create_interactive_sc_t(user_input)
        
        # Determine complexity
        if complexity:
            try:
                query_complexity = QueryComplexity(complexity)
            except ValueError:
                query_complexity = None
        else:
            query_complexity = None
        
        # Display SC_t info
        if self.verbose:
            print(f"🧠 Consciousness Level: f = {sc_t_state['metrics']['f']:.3f}")
            print(f"⚡ Activation: {sc_t_state['E_t']['activation']:.3f}")
            print(f"😊 Emotional State: {sc_t_state['S_t']['emotional_state']}")
            print(f"🎯 Goal: {sc_t_state['G_t']['primary_goal']}")
            print(f"💭 Automatic Thoughts: {len(sc_t_state['A_t'])}")
            print()
        
        # Process query
        start_time = time.time()
        
        try:
            result = await self.manager.process_consciousness_query(
                sc_t_state=sc_t_state,
                user_input=user_input,
                query_complexity=query_complexity
            )
            
            processing_time = time.time() - start_time
            
            # Update session stats
            self.session_stats['queries_processed'] += 1
            self.session_stats['total_processing_time_ms'] += result.processing_time_ms
            if result.success:
                self.session_stats['successful_queries'] += 1
            
            # Display results
            self._display_query_result(result, processing_time)
            
            return {
                'success': result.success,
                'response': result.response,
                'backend_used': result.backend_used,
                'processing_time_ms': result.processing_time_ms,
                'consciousness_integration': result.consciousness_integration
            }
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            if self.debug:
                traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'response': "Sorry, I encountered an error processing your request."
            }
    
    def _display_query_result(self, result, processing_time: float):
        """Display formatted query result"""
        
        # Status line
        status_icon = "✅" if result.success else "❌"
        print(f"{status_icon} Processing {'completed' if result.success else 'failed'}")
        
        # Key metrics
        print(f"🤖 Backend: {result.backend_used}")
        print(f"⏱️  Time: {result.processing_time_ms:.1f}ms")
        
        if result.fallback_used:
            print("🔄 Fallback used")
        
        if result.consciousness_integration:
            ci = result.consciousness_integration
            print(f"🧠 Consciousness: {ci.get('consciousness_level', 'unknown')} (f={ci.get('f_score', 0):.3f})")
        
        # Response
        print(f"\n💬 Response:")
        print("-" * 30)
        print(result.response)
        print("-" * 30)
        
        # Debug info
        if self.debug and result.performance_metrics:
            print(f"\n🔧 Debug Info:")
            pm = result.performance_metrics
            if 'stage_timings' in pm:
                for stage, time_ms in pm['stage_timings'].items():
                    print(f"  {stage}: {time_ms:.1f}ms")
        
        if result.error_message:
            print(f"\n⚠️ Error: {result.error_message}")
    
    async def interactive_mode(self):
        """Run interactive mode"""
        print("🎯 Phase 4 Interactive Mode")
        print("="*40)
        print("Enter queries to test consciousness-enhanced processing.")
        print("Commands:")
        print("  /help     - Show help")
        print("  /status   - Show system status")
        print("  /stats    - Show session statistics")
        print("  /examples - Run example scenarios")
        print("  /quit     - Exit")
        print("="*40)
        
        while True:
            try:
                user_input = input("\n🧠 Query> ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    await self._handle_command(user_input)
                    continue
                
                # Process query
                await self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                if self.debug:
                    traceback.print_exc()
    
    async def _handle_command(self, command: str):
        """Handle CLI commands"""
        
        if command == '/help':
            print("\n📚 Phase 4 CLI Help")
            print("-" * 20)
            print("Commands:")
            print("  /help     - Show this help message")
            print("  /status   - Show system status and hardware info")
            print("  /stats    - Show session statistics")
            print("  /examples - Run example scenarios")
            print("  /debug    - Toggle debug mode")
            print("  /verbose  - Toggle verbose mode")
            print("  /quit     - Exit the CLI")
            print("\nQuery Processing:")
            print("  Just type your question or request normally.")
            print("  The system will automatically create appropriate SC_t states.")
            print("  Use natural language - consciousness integration is automatic.")
            
        elif command == '/status':
            await self._show_system_status()
            
        elif command == '/stats':
            self._show_session_stats()
            
        elif command == '/examples':
            await self._run_examples()
            
        elif command == '/debug':
            self.debug = not self.debug
            level = logging.DEBUG if self.debug else logging.WARNING
            logging.getLogger().setLevel(level)
            print(f"🔧 Debug mode: {'ON' if self.debug else 'OFF'}")
            
        elif command == '/verbose':
            self.verbose = not self.verbose
            if not self.debug:  # Don't override debug level
                level = logging.INFO if self.verbose else logging.WARNING
                logging.getLogger().setLevel(level)
            print(f"📢 Verbose mode: {'ON' if self.verbose else 'OFF'}")
            
        elif command == '/quit':
            print("👋 Exiting...")
            sys.exit(0)
            
        else:
            print(f"❓ Unknown command: {command}")
            print("Type /help for available commands.")
    
    async def _show_system_status(self):
        """Show system status"""
        print("\n🔧 System Status")
        print("-" * 20)
        
        if not self.manager:
            print("❌ Manager not initialized")
            return
        
        status = self.manager.get_system_status()
        
        # Hardware info
        hw = status['hardware']
        print(f"💻 Hardware:")
        print(f"  Premium: {'Yes' if hw['is_premium'] else 'No'}")
        print(f"  RAM: {hw['total_ram_gb']:.1f}GB")
        print(f"  VRAM: {hw['total_vram_gb']:.1f}GB")
        
        # Components
        components = status['components']
        print(f"\n🔧 Components:")
        for comp, active in components.items():
            icon = "✅" if active else "❌"
            print(f"  {icon} {comp}")
        
        # Backend status
        if 'backends' in status:
            print(f"\n🤖 Backends:")
            backends = status['backends']
            if isinstance(backends, dict):
                for backend, details in backends.items():
                    if isinstance(details, dict):
                        status_icon = "✅" if details.get('status') == 'ready' else "⚠️"
                        print(f"  {status_icon} {backend}: {details.get('status', 'unknown')}")
                    else:
                        print(f"  ❓ {backend}: {details}")
        
        # Statistics
        if 'statistics' in status:
            stats = status['statistics']
            print(f"\n📊 Processing Stats:")
            print(f"  Total Queries: {stats.get('total_queries', 0)}")
            print(f"  Success Rate: {stats.get('successful_queries', 0)}/{stats.get('total_queries', 0)}")
            print(f"  Avg Time: {stats.get('avg_processing_time_ms', 0):.1f}ms")
    
    def _show_session_stats(self):
        """Show session statistics"""
        print("\n📊 Session Statistics")
        print("-" * 20)
        
        uptime_seconds = time.time() - self.session_stats['session_start']
        uptime_minutes = uptime_seconds / 60
        
        print(f"⏰ Session Duration: {uptime_minutes:.1f} minutes")
        print(f"🔢 Queries Processed: {self.session_stats['queries_processed']}")
        print(f"✅ Successful Queries: {self.session_stats['successful_queries']}")
        
        if self.session_stats['queries_processed'] > 0:
            success_rate = (self.session_stats['successful_queries'] / self.session_stats['queries_processed']) * 100
            avg_time = self.session_stats['total_processing_time_ms'] / self.session_stats['queries_processed']
            print(f"📈 Success Rate: {success_rate:.1f}%")
            print(f"⏱️  Average Processing Time: {avg_time:.1f}ms")
        
        total_time = self.session_stats['total_processing_time_ms']
        print(f"🕐 Total Processing Time: {total_time:.1f}ms")
    
    async def _run_examples(self):
        """Run example scenarios"""
        print("\n🧪 Running Example Scenarios")
        print("-" * 30)
        
        try:
            runner = Phase4ExampleRunner()
            await runner.initialize()
            
            # Quick examples
            examples = [
                ("Simple Query", lambda: runner.run_basic_integration_example()),
                ("Complex Query", lambda: runner.run_complex_explanation_example()),
                ("Consciousness Query", lambda: runner.run_consciousness_enhanced_example())
            ]
            
            for name, example_func in examples:
                print(f"\n▶️ {name}...")
                try:
                    success = await example_func()
                    print(f"{'✅' if success else '❌'} {name} {'completed' if success else 'failed'}")
                except Exception as e:
                    print(f"❌ {name} failed: {e}")
            
            await runner.cleanup()
            
        except Exception as e:
            print(f"❌ Examples failed: {e}")
    
    async def batch_mode(self, queries: List[str], output_file: Optional[str] = None):
        """Run batch processing mode"""
        print(f"📦 Batch Mode: Processing {len(queries)} queries")
        
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Processing: {query[:50]}...")
            
            result = await self.process_query(query)
            results.append({
                'query': query,
                'timestamp': datetime.now().isoformat(),
                **result
            })
        
        # Save results if output file specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Results saved to: {output_file}")
            except Exception as e:
                print(f"❌ Failed to save results: {e}")
        
        # Summary
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n📊 Batch Summary:")
        print(f"  Processed: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Success Rate: {successful/len(results)*100:.1f}%")
        
        return results


async def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="Phase 4 Layer 3: Consciousness-Enhanced LLM Communication CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive mode
  %(prog)s --batch queries.txt      # Batch process queries from file
  %(prog)s --query "What is consciousness?" # Process single query
  %(prog)s --examples               # Run example scenarios
  %(prog)s --debug --verbose        # Interactive mode with debug output
        """
    )
    
    parser.add_argument('--interactive', '-i', action='store_true', default=True,
                       help='Run in interactive mode (default)')
    parser.add_argument('--query', '-q', type=str,
                       help='Process a single query and exit')
    parser.add_argument('--batch', '-b', type=str,
                       help='Process queries from file (one per line)')
    parser.add_argument('--examples', '-e', action='store_true',
                       help='Run example scenarios')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for batch results (JSON)')
    parser.add_argument('--complexity', '-c', 
                       choices=['simple', 'medium', 'complex', 'consciousness'],
                       help='Override query complexity detection')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--model', '-m', type=str,
                       choices=['gpt-oss', 'mistral', 'mt5', 'api', 'auto'],
                       default='auto',
                       help='Select specific model to use (default: auto)')
    
    args = parser.parse_args()
    
    # Determine mode based on arguments
    if args.query or args.batch or args.examples:
        args.interactive = False
    
    # Create CLI instance
    cli = Phase4CLI(debug=args.debug, verbose=args.verbose, selected_model=args.model)
    
    try:
        # Initialize
        if not await cli.initialize():
            print("❌ Failed to initialize Phase 4 CLI")
            sys.exit(1)
        
        # Run appropriate mode
        if args.examples:
            runner = Phase4ExampleRunner()
            await runner.initialize()
            await runner.run_all_examples()
            await runner.cleanup()
            
        elif args.query:
            result = await cli.process_query(args.query, args.complexity)
            if not result['success']:
                sys.exit(1)
                
        elif args.batch:
            try:
                with open(args.batch, 'r') as f:
                    queries = [line.strip() for line in f if line.strip()]
                await cli.batch_mode(queries, args.output)
            except FileNotFoundError:
                print(f"❌ File not found: {args.batch}")
                sys.exit(1)
                
        else:  # Interactive mode
            await cli.interactive_mode()
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ CLI error: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    asyncio.run(main())