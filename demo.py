#!/usr/bin/env python
"""
RLM Demo Script
===============

Demonstrates using the Recursive Language Model directly.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from core.rlm_engine import RLMEngine
from core.llm_client import LLMClientFactory


async def demo_rlm():
    """Demonstrate RLM with a simple example."""
    
    print("=" * 60)
    print("RLM Demo - Recursive Language Model")
    print("=" * 60)
    
    # Create a large context (simulating a long document)
    context = []
    for i in range(100):
        section = f"""
=== Section {i+1}: The History of Computing ===

In the early days of computing, section {i+1} was particularly important.
The development of early computers relied heavily on the principles outlined here.
Key figures mentioned in section {i+1} include Ada Lovelace and Alan Turing.
The mechanical calculator was a precursor to modern computers.
Section {i+1} also discusses the ENIAC, one of the first electronic computers.
        """
        context.append(section)
    
    print(f"\n[DOC] Created simulated document with {len(context)} sections")
    total_chars = sum(len(s) for s in context)
    print(f"   Total characters: {total_chars:,}")
    print(f"   This would exceed a typical LLM context window!")
    
    # Initialize mock clients (use real ones with your API keys)
    print("\n[AI] Initializing LLM clients (mock mode)...")
    root_client = LLMClientFactory.create("mock", "mock-root")
    sub_client = LLMClientFactory.create("mock", "mock-sub")
    
    # Create RLM Engine
    print("[CONFIG]  Creating RLM Engine...")
    rlm = RLMEngine(
        root_llm_client=root_client,
        sub_llm_client=sub_client,
        max_iterations=10,
        max_repl_output_chars=2000,
        sub_llm_max_chars=50000,
    )
    
    # Query
    query = "What is discussed in section 50?"
    print(f"\n[Q] Query: {query}")
    print("\n[RUN] Running RLM (this may take a moment)...")
    print("-" * 60)
    
    result = await rlm.run(
        query=query,
        context=context,
        context_type="list"
    )
    
    print("\n[RESULTS] Results:")
    print("-" * 60)
    print(f"Success: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Sub-LM calls: {result.get('sub_lm_calls', 0)}")
    
    if result['success']:
        print(f"\n[OK] Answer:")
        print(result['answer'])
    else:
        print(f"\n[ERR] Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nTo use real LLMs, set your API keys:")
    print("  export OPENAI_API_KEY=your-key")
    print("  export RLM_ROOT_PROVIDER=openai")
    print("  export RLM_SUB_PROVIDER=openai")


if __name__ == "__main__":
    asyncio.run(demo_rlm())
