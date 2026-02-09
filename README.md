# RLM - Recursive Language Model

A production-ready implementation of **Recursive Language Models (RLM)** as described in the paper by Zhang et al. (2026). This application enables efficient processing of arbitrarily long documents through recursive LLM retrieval, dramatically outperforming traditional static RAG approaches.

## 🌟 Features

- **🆓 Zero Cost to Deployer**: Users bring their own API keys (BYOK)
- **🆓 Zero Cost to Deployer**: Users bring their own API keys (BYOK)
- **Recursive Language Model (RLM)**: Full implementation of the RLM scaffold from the paper
- **11 LLM Providers**: OpenAI, Anthropic, Groq, Google, Mistral, Cohere, DeepSeek, Together AI, Azure, Ollama (local), Mock
- **REPL Environment**: Prompts loaded as variables with programmatic access
- **Symbolic Recursion**: Code-based decomposition and recursive sub-LM calls
- **Multi-Format Support**: PDF, DOCX, TXT, Markdown, JSON, Code files
- **Intelligent Chunking**: Respects document boundaries (paragraphs, sections, functions)
- **Web Interface**: Modern, responsive UI for document upload and querying
- **REST API**: Full API for integration with other systems
- **Session Management**: Persistent sessions with multiple document support

## 💰 Bring Your Own Key (BYOK)

**Deploy this app for FREE.** Users provide their own OpenAI/Anthropic API keys:

- ✅ **No API costs for you** (the deployer)
- ✅ **No subscription needed**
- ✅ **Keys stored in memory only** (never saved to disk)
- ✅ **Users pay only for what they use** (~$0.10-$0.50 per document)

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (users bring their own - no cost to deployer!)

### Installation

```bash
# Clone or navigate to the project directory
cd rlm_app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

#### Option A: User-Provided Keys (Recommended - Zero Cost to You)
Users enter their API keys in the web UI. Keys are stored in memory only and never saved to disk.

#### Option B: Default Keys (Not Recommended)
Set environment variables as fallback:

```bash
# Server Configuration
export PORT=8000
export HOST="0.0.0.0"

# Optional: Fallback API keys (if you want to provide default keys)
# export OPENAI_API_KEY="your-api-key"
# export ANTHROPIC_API_KEY="your-api-key"
```

### Running the Application

```bash
# Start the server
python -m backend.main

# Or with uvicorn directly
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the web interface at: http://localhost:8000/web

## 📖 How It Works

### The RLM Paradigm

Traditional LLMs are limited by their context window. RLM overcomes this by:

1. **Environment Offloading**: The prompt is loaded as a variable in a REPL environment, NOT in the LLM's context window
2. **Symbolic Manipulation**: The LLM writes Python code to examine, filter, and decompose the prompt
3. **Recursive Sub-Calls**: The LLM can call itself (`llm_query()`) on chunks of the prompt
4. **Iterative Refinement**: Results are aggregated through REPL variables until a final answer is produced

### Comparison with Static RAG

| Feature | Static RAG | RLM |
|---------|-----------|-----|
| Context Handling | Embedding-based retrieval | Symbolic, code-based manipulation |
| Chunk Relevance | Fixed chunking, semantic search | Adaptive, programmatic filtering |
| Multi-hop Reasoning | Limited | Native through recursion |
| Context Window | Bounded by embedding model | Unbounded through recursion |
| Adaptability | Pre-defined | Dynamic, LLM-driven |

## 📚 API Endpoints

### Set API Keys (BYOK)
```bash
POST /api/keys
Content-Type: application/json

{
  "session_id": "uuid",
  "openai_api_key": "sk-...",
  "root_model": "gpt-4o-mini",
  "sub_model": "gpt-4o-mini"
}
```

### Upload Documents
```bash
POST /upload
Content-Type: multipart/form-data

file: <file>
session_id: <optional>
```

### Query with RLM
```bash
POST /query
Content-Type: application/json

{
  "session_id": "uuid",
  "query": "What are the main points?",
  "use_subcalls": true,
  "max_iterations": 50,
  "openai_api_key": "sk-..."  // Optional: can use session keys
}
```

### Get Available Models
```bash
GET /api/models
```

### Get Session Info
```bash
GET /sessions/{session_id}
```

## 🔧 Advanced Configuration

### Custom System Prompts

You can customize the RLM behavior by providing custom system prompts:

```python
from core.rlm_engine import RLMEngine
from core.llm_client import LLMClientFactory

root_client = LLMClientFactory.create("openai", "gpt-4o")
sub_client = LLMClientFactory.create("openai", "gpt-4o-mini")

rlm = RLMEngine(
    root_llm_client=root_client,
    sub_llm_client=sub_client,
    max_iterations=50,
    chunk_size=100000
)

result = await rlm.run(
    query="Your question",
    context=large_document,
    system_prompt="Custom system prompt..."
)
```

### Document Processing Options

```python
from core.document_processor import DocumentProcessor

processor = DocumentProcessor(
    chunk_size=100000,      # Characters per chunk
    chunk_overlap=1000,     # Overlap between chunks
    respect_boundaries=True # Respect paragraph/section boundaries
)
```

## 📊 Performance

Based on the paper's findings:

- **Scalability**: Handles inputs up to 10M+ tokens (2 orders of magnitude beyond context windows)
- **Accuracy**: 28.3% improvement over base models on long-context tasks
- **Cost**: Comparable to base model calls (median), with higher variance
- **Speed**: Depends on trajectory length; can be parallelized

## 🔬 Research Background

This implementation is based on:

**"Recursive Language Models"**  
Alex L. Zhang, Tim Kraska, Omar Khattab  
MIT CSAIL, January 2026

Key innovations:
- Algorithm 1: The RLM scaffold with REPL environment
- Symbolic recursion for unbounded context processing
- Post-training recipe for native RLM behavior

## 🚀 Deployment

### Option 1: Docker (Recommended)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
```

### Option 2: Render.com (Free)

1. Fork/push this repo to GitHub
2. Connect to [Render](https://render.com)
3. Add environment variables:
   - `OPENAI_API_KEY`
4. Deploy!

### Option 3: Railway.app (Free)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

### Option 4: VPS (DigitalOcean, AWS, etc.)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## 🧪 Testing

Run tests:

```bash
pytest
```

Test with mock client (no API costs):

```bash
export RLM_ROOT_PROVIDER="mock"
export RLM_SUB_PROVIDER="mock"
python -m backend.main
```

## 📝 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- Original paper authors: Alex L. Zhang, Tim Kraska, Omar Khattab
- MIT CSAIL for the research

## 📚 Documentation

- [Deployment Guide](DEPLOYMENT.md) - Comprehensive deployment instructions
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical details
