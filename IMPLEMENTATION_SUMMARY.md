# RLM Implementation Summary

## Overview

This is a complete, production-ready implementation of **Recursive Language Models (RLM)** as described in the paper "Recursive Language Models" by Zhang et al. (2026) from MIT CSAIL.

## What is RLM?

RLM is a paradigm that allows LLMs to process arbitrarily long documents by:

1. **Environment Offloading**: Loading the prompt as a variable in a REPL environment (not in the LLM's context window)
2. **Symbolic Manipulation**: The LLM writes Python code to examine, filter, and decompose the prompt
3. **Recursive Sub-Calls**: The LLM can call itself (`llm_query()`) programmatically on chunks
4. **Iterative Refinement**: Results are aggregated through REPL variables until a final answer

## Key Advantages Over Static RAG

| Aspect | Static RAG | RLM |
|--------|-----------|-----|
| Context Handling | Embedding-based retrieval | Symbolic, code-based manipulation |
| Chunk Relevance | Fixed chunking | Adaptive, programmatic filtering |
| Multi-hop Reasoning | Limited | Native through recursion |
| Context Window | Bounded by embedding model | Unbounded through recursion |
| Scalability | ~100K tokens | 10M+ tokens (2 orders of magnitude more) |

## Project Structure

```
rlm_app/
├── backend/
│   └── main.py                 # FastAPI REST API
├── core/
│   ├── __init__.py
│   ├── rlm_engine.py           # Core RLM implementation (Algorithm 1 from paper)
│   ├── llm_client.py           # Unified LLM client (OpenAI, Anthropic, Mock)
│   └── document_processor.py   # Document loading and intelligent chunking
├── frontend/
│   └── index.html              # Web interface
├── tests/
│   ├── __init__.py
│   └── test_rlm_engine.py      # Unit tests
├── uploads/                    # Document storage (created at runtime)
├── venv/                       # Virtual environment
├── .env.example                # Configuration template
├── demo.py                     # Standalone demo
├── start.py                    # Application startup script
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

## Core Components

### 1. RLMEngine (`core/rlm_engine.py`)

Implements Algorithm 1 from the paper:

```python
class RLMEngine:
    def __init__(self, root_llm_client, sub_llm_client, max_iterations=50):
        # Initialize with LLM clients for root and sub-calls
    
    async def run(self, query, context):
        # 1. Initialize REPL with context as variable
        # 2. Inject llm_query() function for recursive calls
        # 3. Iteratively:
        #    - Get code from root LLM
        #    - Execute code in REPL
        #    - Check for FINAL() or FINAL_VAR() answer
        # 4. Return final answer
```

**Key Features:**
- Safe REPL environment with restricted builtins
- Code block extraction and execution
- Final answer extraction (FINAL() and FINAL_VAR())
- Iteration limits for safety
- Metadata feedback to root LLM

### 2. DocumentProcessor (`core/document_processor.py`)

Handles large documents:
- PDF parsing with PyPDF
- DOCX support
- Markdown with header-aware chunking
- Code files with function/class boundary awareness
- Intelligent chunking with configurable overlap

### 3. LLMClient (`core/llm_client.py`)

Unified interface for:
- OpenAI (GPT-4, GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet, etc.)
- Mock client for testing (no API costs)

## Usage

### Quick Start (Web Interface)

```bash
cd rlm_app
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
python start.py
```

Access the web interface at: http://localhost:8000/web

### API Usage

Upload a document:
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload
```

Query with RLM:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id", "query": "What are the main points?"}' \
  http://localhost:8000/query
```

### Direct Usage (Python)

```python
from core.rlm_engine import RLMEngine
from core.llm_client import LLMClientFactory

# Initialize clients
root_client = LLMClientFactory.create("openai", "gpt-4o")
sub_client = LLMClientFactory.create("openai", "gpt-4o-mini")

# Create RLM engine
rlm = RLMEngine(
    root_llm_client=root_client,
    sub_llm_client=sub_client,
    max_iterations=50
)

# Process large document
large_context = "..."  # 10M+ tokens
result = await rlm.run(
    query="Find specific information",
    context=large_context
)

print(result["answer"])
```

## System Prompt

The RLM system prompt (from Appendix C.1 of the paper) instructs the LLM to:

1. Use the REPL environment with `context` variable
2. Call `llm_query()` for recursive sub-calls
3. Use code blocks (```repl) for execution
4. Provide FINAL() or FINAL_VAR() when complete

Example RLM strategy:
```python
# Chunk and query
chunk_size = len(context) // 10
answers = []
for i in range(10):
    chunk = context[i*chunk_size:(i+1)*chunk_size]
    answer = llm_query(f"Analyze: {chunk}")
    answers.append(answer)

# Aggregate
final = llm_query(f"Combine: {answers}")
FINAL_VAR(final)
```

## Configuration

Environment variables:

```bash
# LLM Providers
RLM_ROOT_PROVIDER=openai  # or anthropic, mock
RLM_SUB_PROVIDER=openai
RLM_ROOT_MODEL=gpt-4o
RLM_SUB_MODEL=gpt-4o-mini

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Server
PORT=8000
HOST=0.0.0.0
```

## Testing

```bash
# Run all tests
pytest tests/

# Run demo (mock mode)
python demo.py
```

## Performance Characteristics

Based on the paper and implementation:

- **Scalability**: Handles 10M+ tokens (vs ~128K-272K for base models)
- **Accuracy**: 28.3% improvement over base models on long-context tasks
- **Cost**: Comparable to base model at median, higher variance
- **Speed**: Depends on trajectory length, can be parallelized

## Key Design Decisions

1. **REPL Safety**: Restricted builtins, no file system access
2. **Metadata Feedback**: Only truncated outputs returned to root LLM
3. **Symbolic Recursion**: Code-based loops over verbalized calls
4. **Variable Storage**: Intermediate results in REPL variables
5. **Chunking Strategy**: Respect document boundaries

## Limitations

From the paper (Section 6):
- Synchronous sub-calls (can be parallelized)
- Max recursion depth of 1 (sub-calls are base LMs)
- Distinguishing final answers can be brittle
- Requires coding-capable models

## Future Improvements

- Async sub-call parallelization
- Deeper recursion levels
- Native RLM training
- Streaming responses
- Vector store integration for hybrid approaches

## References

Zhang, A. L., Kraska, T., & Khattab, O. (2026). Recursive Language Models. MIT CSAIL.

Paper: https://github.com/alexzhang13/rlm
