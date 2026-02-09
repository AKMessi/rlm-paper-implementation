# Supported LLM Providers

RLM supports **11 different LLM providers**, giving users flexibility to choose based on cost, speed, and quality.

---

## 📊 Quick Comparison

| Provider | Best For | Cost | Speed | Quality |
|----------|----------|------|-------|---------|
| **DeepSeek** | Cheapest option | $0.07/M | Fast | Good |
| **Groq** | Fast inference | $0.05-0.60/M | ⚡ Very Fast | Good |
| **Google Gemini** | Cheap & fast | $0.08-3.50/M | Fast | Very Good |
| **OpenAI 4o-mini** | Reliable | $0.15-0.60/M | Fast | Very Good |
| **Together AI** | Open source | $0.18-0.90/M | Medium | Good |
| **Mistral** | European provider | $0.20-4/M | Fast | Very Good |
| **Cohere** | Command models | $0.50-2.50/M | Medium | Good |
| **Anthropic** | Best quality | $0.50-15/M | Medium | ⭐ Excellent |
| **OpenAI 4o** | Premium | $2.50-10/M | Fast | ⭐ Excellent |
| **Ollama** | Free local | FREE | Depends on HW | Good |
| **Mock** | Testing | FREE | Instant | N/A |

---

## 🔑 Provider Details

### 1. **OpenAI** (Most Popular)
- **Models**: GPT-4o, GPT-4o-mini, GPT-4-turbo
- **Key URL**: https://platform.openai.com/api-keys
- **Cost**: $0.15-10 per 1M tokens
- **Best For**: Reliability, quality, documentation

```
Root: GPT-4o (best quality)
Sub: GPT-4o-mini (cheap & fast)
```

---

### 2. **Anthropic** (Best Quality)
- **Models**: Claude 3.5 Sonnet, Claude 3 Haiku
- **Key URL**: https://console.anthropic.com/
- **Cost**: $0.50-15 per 1M tokens
- **Best For**: Complex reasoning, long context

```
Root: Claude 3.5 Sonnet (best quality)
Sub: Claude 3 Haiku (faster, cheaper)
```

---

### 3. **Groq** (Fastest)
- **Models**: Llama 3.1 70B, Llama 3.1 8B, Mixtral 8x7B
- **Key URL**: https://console.groq.com/keys
- **Cost**: $0.05-0.60 per 1M tokens
- **Best For**: Speed, cost-effective

```
Key Prefix: gsk_
Root: Llama 3.1 70B
Sub: Llama 3.1 8B (super cheap & fast)
```

---

### 4. **Together AI** (Open Source)
- **Models**: Llama 3.1, Qwen, Mixtral
- **Key URL**: https://api.together.xyz/
- **Cost**: $0.18-0.90 per 1M tokens
- **Best For**: Open source models, variety

---

### 5. **Google Gemini** (Good Free Tier)
- **Models**: Gemini 1.5 Flash, Gemini 1.5 Pro
- **Key URL**: https://aistudio.google.com/app/apikey
- **Cost**: $0.08-3.50 per 1M tokens
- **Best For**: Free tier, Google integration

---

### 6. **Mistral AI** (European)
- **Models**: Mistral Large, Medium, Small
- **Key URL**: https://console.mistral.ai/
- **Cost**: $0.20-4 per 1M tokens
- **Best For**: European data residency

---

### 7. **Cohere** (Command Models)
- **Models**: Command R+, Command R
- **Key URL**: https://dashboard.cohere.com/api-keys
- **Cost**: $0.50-2.50 per 1M tokens
- **Best For**: Enterprise, RAG-focused

---

### 8. **DeepSeek** (Cheapest!)
- **Models**: DeepSeek Chat, DeepSeek Coder
- **Key URL**: https://platform.deepseek.com/
- **Cost**: $0.07-0.30 per 1M tokens ⚡ **CHEAPEST**
- **Best For**: Cost savings, coding

```
Root: DeepSeek Chat ($0.07/M)
Sub: DeepSeek Chat ($0.07/M)
Total: ~$0.05 per document!
```

---

### 9. **Azure OpenAI** (Enterprise)
- **Models**: GPT-4o, GPT-4, GPT-3.5
- **Key URL**: https://portal.azure.com
- **Cost**: Enterprise pricing
- **Best For**: Enterprise, compliance, existing Azure users

---

### 10. **Ollama** (FREE - Local)
- **Models**: Llama 3.1, Mistral, Qwen, Gemma
- **Key URL**: https://ollama.com/
- **Cost**: **FREE** (runs on your machine)
- **Best For**: Privacy, no API costs, offline

```bash
# Install Ollama
# Run locally - completely free!
ollama run llama3.1
```

---

### 11. **Mock** (Testing)
- **Models**: Mock Model
- **Key**: None needed
- **Cost**: FREE
- **Best For**: Testing without API calls

---

## 💰 Cost Comparison (Typical 100-page Document)

| Provider | Root Model | Sub Model | Total Cost |
|----------|------------|-----------|------------|
| **DeepSeek** | DeepSeek ($0.07/M) | DeepSeek ($0.07/M) | ~$0.03 |
| **Groq** | Llama 3.1 70B ($0.59/M) | Llama 3.1 8B ($0.05/M) | ~$0.10 |
| **OpenAI** | GPT-4o-mini ($0.15/M) | GPT-4o-mini ($0.15/M) | ~$0.08 |
| **Google** | Gemini Flash ($0.08/M) | Gemini Flash ($0.08/M) | ~$0.05 |
| **Anthropic** | Claude 3.5 ($3/M) | Claude Haiku ($0.50/M) | ~$1.00 |
| **OpenAI** | GPT-4o ($2.50/M) | GPT-4o-mini ($0.15/M) | ~$0.80 |
| **Ollama** | Llama 3.1 (FREE) | Llama 3.1 (FREE) | **$0.00** |

---

## ⭐ Recommended Configurations

### 🏆 Best Bang for Buck
```
Root: DeepSeek Chat
Sub: DeepSeek Chat
Cost: ~$0.03 per document
```

### ⚡ Fastest Processing
```
Root: Groq Llama 3.1 70B
Sub: Groq Llama 3.1 8B
Speed: ~2-5 seconds per query
```

### 🎯 Best Quality
```
Root: GPT-4o or Claude 3.5 Sonnet
Sub: GPT-4o-mini or Claude Haiku
Quality: Excellent
```

### 💸 Completely Free
```
Root: Ollama Llama 3.1
Sub: Ollama Llama 3.1
Cost: $0 (requires local setup)
```

### 🧪 Testing
```
Root: Mock
Sub: Mock
Cost: $0 (fake responses)
```

---

## 🔧 Setting Up Each Provider

### OpenAI
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Key format: `sk-...`

### Anthropic
1. Go to https://console.anthropic.com/
2. Get API key
3. Key format: `sk-ant-...`

### Groq
1. Go to https://console.groq.com/keys
2. Create API key
3. Key format: `gsk_...`

### DeepSeek
1. Go to https://platform.deepseek.com/
2. Register and get API key
3. Key format: `sk-...`

### Google Gemini
1. Go to https://aistudio.google.com/app/apikey
2. Create API key
3. No specific prefix

### Ollama (Local)
1. Install from https://ollama.com/
2. Run: `ollama pull llama3.1`
3. Run: `ollama serve`
4. No API key needed!

---

## 🚀 Pro Tips

1. **Use DeepSeek for cost savings** - cheapest option, good quality
2. **Use Groq for speed** - incredibly fast inference
3. **Use GPT-4o-mini as sub-model** - cheap and reliable
4. **Use Ollama for privacy** - everything stays local
5. **Mix providers** - expensive model for root, cheap for sub-calls

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Invalid API key" | Check key prefix matches provider |
| "Rate limit" | Use a different provider or wait |
| "Model not found" | Check provider supports that model |
| "Ollama connection refused" | Make sure Ollama is running: `ollama serve` |
| "Too slow" | Switch to Groq or use GPT-4o-mini |

---

## 📈 Adding New Providers

To add a new provider:

1. Add client class in `core/llm_client.py`
2. Add to `LLMClientFactory`
3. Update `backend/main.py` API key handling
4. Add to frontend dropdown
5. Update this documentation

---

**Bottom Line: You have 11 options ranging from FREE to premium. Choose based on your budget and quality needs!**
