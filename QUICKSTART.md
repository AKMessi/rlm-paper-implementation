# RLM Quick Start Guide

Get your RLM application running in 5 minutes!

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
3. Add your `OPENAI_API_KEY` in dashboard
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

## 🔑 Required Environment Variables

Create a `.env` file:

```bash
# For real LLM usage (required for production)
OPENAI_API_KEY=sk-your-key-here
RLM_ROOT_PROVIDER=openai
RLM_SUB_PROVIDER=openai

# For testing (no API costs)
RLM_ROOT_PROVIDER=mock
RLM_SUB_PROVIDER=mock
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
| API errors | Check OPENAI_API_KEY is set correctly |
| Upload fails | Ensure `uploads/` directory exists |

---

## 📚 Next Steps

1. **Read the paper**: [Recursive Language Models](https://github.com/alexzhang13/rlm)
2. **Try the API**: See [README.md](README.md) for API examples
3. **Customize**: Modify system prompts in `core/rlm_engine.py`
4. **Scale**: Add Redis, multiple workers for production

---

**Need help?** Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.
