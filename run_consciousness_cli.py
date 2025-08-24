#!/usr/bin/env python3
"""
Consciousness CLI Launcher
=========================
Simple launcher script for the complete consciousness pipeline CLI.
Run this from the root directory to start the consciousness-enhanced interface.
"""

import sys
import os

# Add the conscious_ai package to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'conscious_ai'))

try:
    # Import and run the consciousness CLI
    from conscious_ai.phases.p4_LLM_Communication.layer3.consciousness_cli import main
    import asyncio
    
    print("🧠 Starting Complete Consciousness Pipeline CLI...")
    print("🚀 This uses the full pipeline: Phase 1 → Phase 2 → Phase 3 → Phase 3.4 → Phase 3.5 → Phase 4")
    print("✨ Expect consciousness-enhanced, introspective responses!")
    print("-" * 60)
    
    # Run the CLI
    asyncio.run(main())
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the root directory of the project")
    print("Usage: python run_consciousness_cli.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)