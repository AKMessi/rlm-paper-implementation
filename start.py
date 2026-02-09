#!/usr/bin/env python
"""
RLM Application Startup Script
==============================

Simple startup script with environment checks and helpful messages.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print startup banner."""
    print("""
===============================================================

              Recursive Language Model (RLM)

          Process arbitrarily long documents using
              recursive LLM retrieval

===============================================================
    """)

def check_environment():
    """Check if the environment is properly configured."""
    issues = []
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    root_provider = os.getenv("RLM_ROOT_PROVIDER", "mock")
    
    if root_provider == "openai" and not openai_key:
        issues.append("OPENAI_API_KEY not set (required for OpenAI provider)")
    
    if root_provider == "anthropic" and not anthropic_key:
        issues.append("ANTHROPIC_API_KEY not set (required for Anthropic provider)")
    
    if root_provider == "mock":
        print("[!] Running in MOCK mode (no real LLM calls)")
        print("   Set RLM_ROOT_PROVIDER=openai to use real LLMs\n")
    
    return issues

def main():
    """Main startup function."""
    print_banner()
    
    # Check environment
    issues = check_environment()
    
    if issues:
        print("[!] Environment Issues:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nYou can still run in mock mode by setting RLM_ROOT_PROVIDER=mock")
        print("Or set the required API keys and try again.\n")
        
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            print("Exiting...")
            sys.exit(1)
    
    # Check if venv is activated
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv and Path("venv").exists():
        print("[TIP] You may want to activate your virtual environment:")
        print("   Windows: venv\\Scripts\\activate")
        print("   macOS/Linux: source venv/bin/activate\n")
    
    # Get configuration
    port = os.getenv("PORT", "8000")
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"[START] Starting RLM Server on http://{host}:{port}")
    print(f"   Web Interface: http://localhost:{port}/web\n")
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=int(port),
            reload=True,
            log_level="info"
        )
    except ImportError:
        print("[ERROR] uvicorn not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn[standard]"])
        print("\n[OK] Installed! Please run again.")
    except KeyboardInterrupt:
        print("\n\n[BYE] Shutting down RLM server...")

if __name__ == "__main__":
    main()
