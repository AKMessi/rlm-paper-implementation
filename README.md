# 🔄 RLM - Recursive Language Model

> True implementation of **"Recursive Language Models"** from MIT  
> Process 10M+ token documents through symbolic recursion

[![Tests](https://img.shields.io/badge/tests-16%2F16%20passing-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 🤯 What is this?

This is a **true implementation** of [Algorithm 1](RLM_IMPLEMENTATION.md) from the paper *"Recursive Language Models"* by Zhang et al. (MIT CSAIL, 2026).

**The Problem:** LLMs have limited context windows (~128K tokens). Traditional RAG uses embeddings that miss nuanced connections.

**The Solution:** RLM treats the prompt as a **REPL environment variable**. The LLM writes Python code to:
1. 🔍 Examine the context programmatically  
2. 🔄 Call itself recursively via `llm_query()` on chunks
3. 🧠 Build up answers through symbolic manipulation
4. ✅ Set `Final` variable when done

**Result:** Handles 10M+ tokens (2 orders of magnitude beyond context limits).

---

## ✨ Features

- 📄 **10M+ Token Support** - Process arbitrarily long documents
- 🎯 **True RLM Algorithm** - Exact implementation of Algorithm 1 from paper
- 🔑 **BYOK** - Users bring their own API keys (zero cost to deployer)
- 🤖 **15+ LLM Providers** - OpenAI, Anthropic, Google, Groq, Together, Mistral, Cohere, DeepSeek, Perplexity, Azure, Ollama
- 📑 **Multi-Format** - PDF, DOCX, TXT, Markdown, JSON, Code files
- 🌐 **Web UI** - Modern interface for upload & chat
- ⚡ **FastAPI Backend** - REST API for integration

---

## 🎬 Demo

**Live Demo:** https://rlm-ucnx.onrender.com/web

![Demo](https://img.shields.io/badge/🚀-Try%20it%20now-blue)

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Prompt   │────▶│  REPL Environment │────▶│  LLM Generates │
│   (10M tokens)  │     │  context = P      │     │  Python Code   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Return Final   │◀────│ Check Final var │◀────│  Execute Code   │
│     Answer      │     │  in REPL state  │     │  in REPL        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌─────────────────┐
                        │ llm_query() for │
                        │ recursive calls │
                        └─────────────────┘
```

---

## 🚀 Quick Start

```bash
# Clone & setup
git clone https://github.com/AKMessi/rlm-paper-implementation.git
cd rlm-paper-implementation/rlm_app

# Create venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Run
python -m backend.main
```

Open http://localhost:8000/web

---

## 📖 How It Works

### Traditional RAG
```
Document → Chunks → Embeddings → Vector DB → Similarity Search → LLM
```
❌ Static, loses nuance, embedding bottleneck

### RLM (This Implementation)
```
Document → REPL Variable → LLM Writes Code → Code Examines Doc
                ↓
        llm_query() chunks → Aggregate → Set Final Variable
```
✅ Dynamic, programmable, unbounded context

---

## 📡 API Usage

### 1. Set API Keys (BYOK)
```bash
POST /api/keys
{
  "session_id": "your-session-id",
  "openai_api_key": "sk-...",
  "root_model": "gpt-4o",
  "sub_model": "gpt-4o-mini"
}
```

### 2. Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

file: @your-document.pdf
session_id: your-session-id
```

### 3. Query with RLM
```bash
POST /query
{
  "session_id": "your-session-id",
  "query": "What are the key findings?",
  "max_iterations": 10
}
```

**Response:**
```json
{
  "success": true,
  "answer": "The key findings are...",
  "iterations": 3,
  "sub_lm_calls": 5,
  "processing_time_seconds": 12.5
}
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# 16 tests covering:
# - REPL state management
# - Code execution
# - Algorithm 1 loop
# - Final variable termination
```

---

## 🏛️ Paper Reference

**"Recursive Language Models"**  
Alex L. Zhang, Tim Kraska, Omar Khattab  
MIT CSAIL, January 2026

**Key Innovation:** Algorithm 1 - RLM scaffold with:
- ✓ Prompt as REPL variable (not context window)
- ✓ LLM generates code (not chat)
- ✓ Symbolic recursion via `llm_query()`
- ✓ Termination via `Final` variable
- ✓ Constant-size history

[Read full implementation details →](RLM_IMPLEMENTATION.md)

---

## 🤝 Credits

- **Paper Authors:** [@a1zhang](https://twitter.com/a1zhang) & [@lateinteraction](https://twitter.com/lateinteraction) (Omar Khattab)
- **Implementation:** [@AKMessi](https://github.com/AKMessi)
- **Institution:** MIT CSAIL

---

## 📜 License

MIT License - Feel free to use, modify, deploy!

---

## 🌟 Star History

If you find this useful, please ⭐ star the repo!

[![Star History Chart](https://img.shields.io/github/stars/AKMessi/rlm-paper-implementation?style=social)]()

---

<p align="center">
  <b>🔄 Process the impossible. Recursively.</b>
</p>
