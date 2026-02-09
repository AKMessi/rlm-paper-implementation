# RLM Quick Start Guide

Get your RLM application running in 5 minutes - **with zero API costs to you!**

> 🆓 **BYOK (Bring Your Own Key)**: Users enter their own OpenAI/Anthropic API keys in the UI. You pay nothing!

---

## 💰 How BYOK Works

1. **You deploy** the app (completely free)
2. **Users enter** their own API keys in the web interface
3. **Keys are stored** in memory only (never saved to disk)
4. **Users pay** only for what they use (~$0.10-$0.50 per document)
5. **You pay** $0.00 - forever!

---

## 🎯 Fastest: Run Locally (No Deployment)

```bash
cd rlm_app

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python start.py

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python start.py
```

Open: http://localhost:8000/web

---

## 🚀 One-Click Deployments

### Render.com (Recommended Free Option)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Fork this repo to your GitHub
2. Click the button above
3. **No API keys needed** - users bring their own!
4. Done! 🎉

### Railway.app

```bash
# Install CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

---

## 🐳 Docker (Production Ready)

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📋 Platform Decision Matrix

| Platform | Cost | Difficulty | Best For |
|----------|------|------------|----------|
| **Local** | Free | ⭐⭐ Easy | Development |
| **Render** | Free | ⭐⭐⭐ Easiest | Quick demo |
| **Railway** | $5 credit | ⭐⭐⭐ Easiest | Modern stack |
| **Docker** | Varies | ⭐⭐ Medium | Production |
| **VPS** | $5/mo | ⭐⭐ Hard | Full control |

---

## ⚡ 30-Second Deploy

### Using the deploy script:

```bash
# Make script executable (Unix/Mac)
chmod +x deploy.sh

# Deploy!
./deploy.sh render   # or railway, docker, local
```

---

## 🔑 Environment Variables (Optional)

**No API keys required!** Users enter their own in the UI.

Optional `.env` file for configuration:

```bash
# Server settings
PORT=8000
HOST=0.0.0.0

# Optional: Fallback API keys (not recommended)
# Only use if you want to provide default keys
# OPENAI_API_KEY=sk-your-key-here

# Optional: Use mock mode for testing
# RLM_ROOT_PROVIDER=mock
# RLM_SUB_PROVIDER=mock
```

---

## ✅ Verify Deployment

Once deployed, check:

```bash
# Health check
curl https://your-app-url.com/health

# Should return:
# {"status": "healthy", "active_sessions": 0}
```

Then open the web interface and upload a document!

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Change PORT in .env or stop other services |
| Module not found | Activate venv: `source venv/bin/activate` |
| API errors | User needs to enter their API key in the UI |
| "No API keys set" | User must click "Save API Keys" before querying |
| Upload fails | Ensure `uploads/` directory exists |

---

## 📚 Next Steps

1. **Read the paper**: [Recursive Language Models](https://github.com/alexzhang13/rlm)
2. **Try the API**: See [README.md](README.md) for API examples
3. **Customize**: Modify system prompts in `core/rlm_engine.py`
4. **Scale**: Add Redis, multiple workers for production

---

**Need help?** Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.
