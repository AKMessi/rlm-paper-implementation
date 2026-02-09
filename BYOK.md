# Bring Your Own Key (BYOK)

**Zero cost deployment. Users pay only for what they use.**

---

## What is BYOK?

**Bring Your Own Key** means users provide their own API keys (OpenAI, Anthropic, etc.) instead of the deployer paying for all API usage.

## Why BYOK?

| Problem | BYOK Solution |
|---------|--------------|
| API costs scale with users | Each user pays for their own usage |
| Unexpected bills | No surprises - users control their spending |
| Key security concerns | Keys never touch your infrastructure permanently |
| Subscription complexity | No accounts or subscriptions needed for deployer |

---

## How It Works

### User Flow:

1. **User visits** your deployed app
2. **User enters** their OpenAI/Anthropic API key in the UI
3. **Key is stored** in memory only (session-based, not persisted)
4. **User uploads** documents and queries
5. **User pays** OpenAI/Anthropic directly for API usage
6. **You pay** $0.00

### Technical Details:

- Keys are stored in the **session object** (in-memory dictionary)
- Sessions expire when the server restarts
- Keys are **never written to disk** or database
- Keys are **masked in the UI** after entry (showing only first 8 and last 4 chars)

---

## Security

### For Deployers:
- ✅ No API keys stored in environment variables (optional)
- ✅ No database with sensitive data
- ✅ No liability for API usage costs
- ✅ No risk of key leakage from your infrastructure

### For Users:
- ✅ Keys are transmitted securely (HTTPS)
- ✅ Keys are stored in memory only
- ✅ Keys disappear when session ends
- ✅ Users can revoke keys anytime via OpenAI/Anthropic dashboard

---

## User Instructions

Add this to your README or website:

```markdown
## How to Use

1. **Get an API key** from [OpenAI](https://platform.openai.com/api-keys) or [Anthropic](https://console.anthropic.com/)
   - New users get free credits
   - Pay-as-you-go: Only pay for what you use

2. **Enter your key** in the app
   - Click "API Configuration" on the left
   - Paste your key (starts with `sk-`)
   - Click "Save API Keys"

3. **Upload documents** and start querying!

### Estimated Costs:
- Small document (~10 pages): ~$0.05
- Medium document (~50 pages): ~$0.15
- Large document (~200 pages): ~$0.50

Using GPT-4o Mini reduces costs by ~90%!
```

---

## Cost Comparison

### Traditional Deployment (You Pay):
```
100 users × 10 queries/day × $0.10/query × 30 days = $3,000/month
```

### BYOK Deployment (Users Pay):
```
You pay: $0/month (hosting only, often free)
Each user pays: ~$0.10-$0.50 per session
```

---

## Optional: Fallback Keys

If you want to provide default keys (not recommended for public deployments):

```bash
# .env file
OPENAI_API_KEY=sk-your-key-here
```

Users can still override with their own keys in the UI.

---

## Disabling BYOK (Not Recommended)

To require environment variable keys only:

1. Modify `backend/main.py` to remove the user key option
2. Remove the API key UI from `frontend/index.html`

But seriously, don't do this. BYOK is better for everyone.

---

## FAQ

**Q: Can users see each other's keys?**  
A: No. Keys are stored per-session in memory only.

**Q: What happens if a user enters a wrong key?**  
A: They get an error message and can try again. No cost incurred.

**Q: Can I see how much users are spending?**  
A: No, and you shouldn't want to. Each user pays OpenAI/Anthropic directly.

**Q: Is this against OpenAI's terms?**  
A: No. Users are using their own keys for their own purposes.

**Q: What if a user doesn't have an API key?**  
A: They can get one for free at openai.com (new users get $5-18 credits)

---

## Best Practices

1. **Make it clear** users need their own keys
2. **Provide instructions** on how to get keys
3. **Show cost estimates** so users know what to expect
4. **Recommend cheaper models** (GPT-4o Mini) for cost-conscious users
5. **Add a warning** that keys are stored in memory only

---

**Bottom line: Deploy this app for free. Let users bring their own keys. Sleep well knowing you won't get a surprise $1000 API bill.**
