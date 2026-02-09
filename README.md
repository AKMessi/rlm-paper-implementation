# RLM - Recursive Language Model

True implementation of **Recursive Language Models** from the paper by Zhang et al. (2026). Process arbitrarily long documents through recursive LLM retrieval.

## Features

- **True RLM Algorithm**: Exact implementation of Algorithm 1 from the paper
- **BYOK**: Users bring their own API keys - zero cost to deployer
- **15+ LLM Providers**: OpenAI, Anthropic, Google, Groq, Together, Mistral, Cohere, DeepSeek, Perplexity, Azure, Ollama, etc.
- **Multi-Format**: PDF, DOCX, TXT, MD, JSON, code files
- **Web UI**: Modern interface for document upload and querying
- **REST API**: Full API for integration

## Quick Start

```bash
cd rlm_app
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
python -m backend.main
```

Access at: http://localhost:8000/web

## How RLM Works

1. **Environment Offloading**: Prompt loaded as REPL variable, not in context window
2. **Symbolic Manipulation**: LLM writes Python code to examine/decompose prompt
3. **Recursive Sub-Calls**: `llm_query()` function for chunk processing
4. **Termination**: Loop ends when `Final` variable is set

## API Usage

### Set API Keys
```bash
POST /api/keys
{
  "session_id": "uuid",
  "openai_api_key": "sk-...",
  "root_model": "gpt-4o",
  "sub_model": "gpt-4o-mini"
}
```

### Upload Document
```bash
POST /upload
Content-Type: multipart/form-data
file: <document.pdf>
```

### Query
```bash
POST /query
{
  "session_id": "uuid",
  "query": "What are the main points?",
  "max_iterations": 10
}
```

## Testing

```bash
python -m pytest tests/ -v
```

## Documentation

- `RLM_IMPLEMENTATION.md` - Technical details of Algorithm 1 implementation
- Paper: "Recursive Language Models" by Zhang et al. (2026)

## License

MIT License
