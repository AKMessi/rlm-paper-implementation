# RLM Deployment Guide

Complete deployment instructions for various platforms.

---

## 📋 Prerequisites

Before deploying, ensure you have:
- OpenAI API key (or Anthropic)
- Git repository initialized
- Python 3.10+ available

---

## 🐳 Option 1: Docker Deployment (Recommended)

### Step 1: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Create docker-compose.yml

```yaml
version: '3.8'

services:
  rlm-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - RLM_ROOT_PROVIDER=openai
      - RLM_SUB_PROVIDER=openai
      - RLM_ROOT_MODEL=gpt-4o
      - RLM_SUB_MODEL=gpt-4o-mini
    volumes:
      - ./uploads:/app/uploads
    restart: unless-stopped
```

### Step 3: Deploy

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## ☁️ Option 2: Render.com (Free Tier)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial RLM implementation"
git remote add origin https://github.com/yourusername/rlm-document-retrieval.git
git push -u origin main
```

### Step 2: Create render.yaml

```yaml
# render.yaml
services:
  - type: web
    name: rlm-app
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false  # Set in Render dashboard
      - key: RLM_ROOT_PROVIDER
        value: openai
      - key: RLM_SUB_PROVIDER
        value: openai
      - key: RLM_ROOT_MODEL
        value: gpt-4o
      - key: RLM_SUB_MODEL
        value: gpt-4o-mini
      - key: PYTHON_VERSION
        value: 3.11.0
    disk:
      name: uploads
      mountPath: /app/uploads
      sizeGB: 1
```

### Step 3: Deploy on Render

1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Blueprint"
3. Connect your GitHub repo
4. Add environment variables in dashboard:
   - `OPENAI_API_KEY`: your-key-here
5. Click "Apply"

**URL**: `https://rlm-app.onrender.com`

---

## 🚂 Option 3: Railway.app (Free Tier)

### Step 1: Create railway.json

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn backend.main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Step 2: Deploy

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Add environment variables
railway variables set OPENAI_API_KEY=your-key-here
railway variables set RLM_ROOT_PROVIDER=openai
railway variables set RLM_SUB_PROVIDER=openai

# Deploy
railway up

# Open in browser
railway open
```

---

## 🟣 Option 4: Heroku

### Step 1: Create Procfile

```
web: uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Step 2: Create runtime.txt

```
python-3.11.6
```

### Step 3: Deploy

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create your-rlm-app

# Add environment variables
heroku config:set OPENAI_API_KEY=your-key-here
heroku config:set RLM_ROOT_PROVIDER=openai
heroku config:set RLM_SUB_PROVIDER=openai

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1

# View logs
heroku logs --tail
```

**URL**: `https://your-rlm-app.herokuapp.com`

---

## 🖥️ Option 5: VPS / Cloud Server (AWS, DigitalOcean, Linode)

### Step 1: Setup Server (Ubuntu)

```bash
# SSH into server
ssh user@your-server-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv nginx -y

# Create app directory
mkdir -p /var/www/rlm-app
cd /var/www/rlm-app

# Clone repository
git clone https://github.com/yourusername/rlm-document-retrieval.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create uploads directory
mkdir -p uploads
```

### Step 2: Create Systemd Service

```bash
sudo nano /etc/systemd/system/rlm-app.service
```

Add:
```ini
[Unit]
Description=RLM Application
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/var/www/rlm-app
Environment="PATH=/var/www/rlm-app/venv/bin"
Environment="OPENAI_API_KEY=your-key-here"
Environment="RLM_ROOT_PROVIDER=openai"
Environment="RLM_SUB_PROVIDER=openai"
ExecStart=/var/www/rlm-app/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable rlm-app
sudo systemctl start rlm-app
sudo systemctl status rlm-app
```

### Step 3: Setup Nginx (Reverse Proxy)

```bash
sudo nano /etc/nginx/sites-available/rlm-app
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    location /uploads {
        alias /var/www/rlm-app/uploads;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/rlm-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 4: SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

---

## 🔷 Option 6: AWS (EC2 + Elastic Beanstalk)

### Using Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.11 rlm-app

# Create environment
eb create rlm-app-env

# Set environment variables
eb setenv OPENAI_API_KEY=your-key-here RLM_ROOT_PROVIDER=openai

# Deploy
eb deploy

# Open
eb open
```

### Using EC2 Directly

1. Launch EC2 instance (Ubuntu 22.04, t2.micro or larger)
2. Configure Security Group to allow ports 22 (SSH) and 80/443 (HTTP/HTTPS)
3. Follow VPS deployment steps above

---

## 🔶 Option 7: Google Cloud Platform (Cloud Run)

### Step 1: Create Dockerfile (same as Option 1)

### Step 2: Deploy

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable services
gcloud services enable run.googleapis.com

# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/rlm-app

gcloud run deploy rlm-app \
  --image gcr.io/YOUR_PROJECT_ID/rlm-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "OPENAI_API_KEY=your-key-here,RLM_ROOT_PROVIDER=openai,RLM_SUB_PROVIDER=openai"
```

---

## 📁 Option 8: Local Network / Raspberry Pi

```bash
# On Raspberry Pi or local machine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with host binding
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Access from other devices on network
# http://your-local-ip:8000
```

---

## 🔐 Environment Variables for Production

Create `.env` file (never commit this!):

```bash
# LLM Configuration
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
RLM_ROOT_PROVIDER=openai
RLM_SUB_PROVIDER=openai
RLM_ROOT_MODEL=gpt-4o
RLM_SUB_MODEL=gpt-4o-mini

# Server Configuration
PORT=8000
HOST=0.0.0.0

# Security (add for production)
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
MAX_UPLOAD_SIZE=52428800  # 50MB

# Optional: For rate limiting
RATE_LIMIT_PER_MINUTE=30
```

---

## 📊 Platform Comparison

| Platform | Free Tier | Ease | Best For |
|----------|-----------|------|----------|
| **Render** | ✅ Yes | ⭐⭐⭐ Easy | Quick deployment, hobby projects |
| **Railway** | ✅ $5 credit | ⭐⭐⭐ Easy | Modern apps, good DX |
| **Heroku** | ⚠️ Limited | ⭐⭐⭐ Easy | Simple apps, but pricey |
| **AWS** | ✅ 12 months | ⭐⭐ Complex | Enterprise, scale |
| **GCP** | ✅ $300 credit | ⭐⭐ Complex | ML workloads |
| **DigitalOcean** | ❌ No | ⭐⭐ Medium | Affordable VPS |
| **Docker** | ✅ Free | ⭐⭐ Medium | Portability |

---

## 🚀 Quick Deploy Commands

### Fastest (Render.com)
```bash
git push origin main
# Then connect repo on Render dashboard
```

### Most Control (Docker)
```bash
docker-compose up -d
```

### Cheapest (VPS)
```bash
# $5/month DigitalOcean droplet
# Follow Option 5 above
```

---

## 🔄 Continuous Deployment

### GitHub Actions → Render

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Render

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Render
      env:
        RENDER_API_KEY: ${{ secrets.RENDER_API_KEY }}
      run: |
        curl -X POST \
          https://api.render.com/v1/services/${{ secrets.RENDER_SERVICE_ID }}/deploys \
          -H "Authorization: Bearer $RENDER_API_KEY"
```

---

## ⚡ Performance Tuning

### For High Traffic

1. **Use Gunicorn with Uvicorn workers**:
```bash
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

2. **Add Redis for session storage** (instead of in-memory)

3. **Enable Caching**:
```python
# backend/main.py
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="rlm-cache")
```

4. **Use CDN for static files** (Cloudflare, etc.)

---

## 🛡️ Security Checklist

- [ ] Use HTTPS (SSL certificate)
- [ ] Set strong `SECRET_KEY`
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Set upload size limits
- [ ] Add rate limiting
- [ ] Use environment variables for secrets
- [ ] Regular dependency updates
- [ ] Enable CORS properly (don't use `*` in production)

---

## 🆘 Troubleshooting

### Port already in use
```bash
# Find process using port 8000
sudo lsof -i :8000
# Kill it
kill -9 <PID>
```

### Out of memory
```bash
# Add swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Module not found
```bash
# Ensure you're in virtual environment
which python
# Should show venv path
```

---

## 📞 Need Help?

- **Docker issues**: Check `docker-compose logs`
- **Nginx errors**: `sudo nginx -t` then `sudo tail -f /var/log/nginx/error.log`
- **App crashes**: `sudo journalctl -u rlm-app -f`

---

**Recommendation**: Start with **Render.com** for quick deployment, then move to **Docker on VPS** for production scale.
