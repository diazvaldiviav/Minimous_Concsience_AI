#!/usr/bin/env python3
"""
Consciousness API Server Launcher
=================================
Launches the consciousness API server with proper configuration.

Usage:
    python run_server.py
    python run_server.py --port 8080 --host 127.0.0.1
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False


def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing.append("uvicorn")
    
    try:
        from openai import AsyncOpenAI
    except ImportError:
        missing.append("openai")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nInstall with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True


def check_environment():
    """Check if environment is properly configured"""
    warnings = []
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        warnings.append("OPENAI_API_KEY not set - Phase 7 will be disabled")
    
    # Check if debug directory exists
    debug_dir = project_root / "debug"
    if not debug_dir.exists():
        debug_dir.mkdir(exist_ok=True)
        print(f"📁 Created debug directory: {debug_dir}")
    
    if warnings:
        print("⚠️ Environment warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print()


def main():
    """Main server launcher"""
    parser = argparse.ArgumentParser(
        description="Consciousness API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run on default host:port (0.0.0.0:8000)
  %(prog)s --port 8080              # Run on custom port
  %(prog)s --host 127.0.0.1         # Run on localhost only
  %(prog)s --debug                  # Enable debug mode
  %(prog)s --no-reload              # Disable auto-reload
        """
    )
    
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to (default: 8000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-reload', action='store_true',
                       help='Disable auto-reload')
    parser.add_argument('--log-level', default='info',
                       choices=['debug', 'info', 'warning', 'error'],
                       help='Log level (default: info)')
    
    args = parser.parse_args()
    
    print("🧠 Consciousness API Server")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    check_environment()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Server configuration
    config = {
        "app": "conscious_ai.api.consciousness_endpoint:app",
        "host": args.host,
        "port": args.port,
        "reload": not args.no_reload,
        "log_level": args.log_level,
        "access_log": True,
    }
    
    print(f"🚀 Starting server on http://{args.host}:{args.port}")
    print(f"📚 API documentation: http://{args.host}:{args.port}/docs")
    print(f"📖 ReDoc documentation: http://{args.host}:{args.port}/redoc")
    
    if args.debug:
        print("🐛 Debug mode enabled")
    
    if not args.no_reload:
        print("🔄 Auto-reload enabled")
    
    print("-" * 40)
    
    try:
        # Start server
        uvicorn.run(**config)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()