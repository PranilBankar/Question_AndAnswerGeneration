# 🚀 Deployment Guide — Railway (Backend) + Vercel (Frontend)

## ⚠️ Key Constraint: BERT Model Size
The trained BERT model is **418 MB** (`model.safetensors`).  
GitHub blocks files >100MB. We use **Git LFS** to handle this.

---

## PART 1: Setup Git LFS + Push to GitHub

### Step 1 — Install Git LFS
Download and install: https://git-lfs.com  
Then in your terminal:
```bash
git lfs install
```

### Step 2 — Track large model file
```bash
cd "d:\Users\Pranil\Github Repos\Question_AndAnswerGeneration"
git lfs track "Classification/checkpoints/best_model/model.safetensors"
git add .gitattributes
```

### Step 3 — Stage and commit everything
```bash
git add .
git commit -m "Add deployment files + BERT model via LFS"
git push origin main
```
> ⏳ This will take a few minutes — the 418MB file uploads via LFS.

---

## PART 2: Deploy Backend to Railway

### Step 4 — Create Railway account
Go to: https://railway.app  
Sign up with **GitHub** account (easiest — auto-imports repos)

### Step 5 — Create new project
1. Click **"New Project"**
2. Choose **"Deploy from GitHub repo"**
3. Select your `Question_AndAnswerGeneration` repo
4. Railway auto-detects Python → click **"Deploy"**

### Step 6 — Set environment variables
In your Railway project → **Variables** tab → add these:

| Variable | Value |
|---|---|
| `SUPABASE_URL` | `https://your-project.supabase.co` |
| `SUPABASE_KEY` | `your-supabase-anon-key` |
| `GROQ_API_KEY` | `your-groq-api-key` |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |

### Step 7 — Get your backend URL
After deployment completes (3–8 mins):  
Go to **Settings → Domains** → Copy your URL  
It looks like: `https://question-andanswer-production.up.railway.app`

### Step 8 — Verify backend is live
Open in browser:
```
https://YOUR-RAILWAY-URL.up.railway.app/health
https://YOUR-RAILWAY-URL.up.railway.app/docs
```

---

## PART 3: Update Frontend with Backend URL

### Step 9 — Update API_BASE in app.js
Open `Frontend/app.js` and change line 14:
```javascript
// Before (local):
const API_BASE = "http://localhost:8000";

// After (production):
const API_BASE = "https://YOUR-RAILWAY-URL.up.railway.app";
```

### Step 10 — Commit the frontend update
```bash
git add Frontend/app.js
git commit -m "Update API_BASE to Railway URL"
git push origin main
```

---

## PART 4: Deploy Frontend to Vercel

### Step 11 — Create Vercel account
Go to: https://vercel.com  
Sign up with **GitHub** account

### Step 12 — Import project
1. Click **"New Project"**
2. Import your `Question_AndAnswerGeneration` repo
3. In **"Configure Project"**:
   - **Root Directory**: `Frontend`
   - **Framework Preset**: `Other` (it's vanilla HTML)
   - Leave everything else default
4. Click **"Deploy"**

### Step 13 — Get your frontend URL
Vercel gives you: `https://question-andanswer.vercel.app`

---

## ✅ Final Test

1. Open your Vercel URL
2. Type a question: *"What is the role of cofactors in enzyme activity?"*
3. Click **🔬 Full Analysis**
4. Should classify → retrieve → answer in ~10 seconds

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| Railway deploy fails | Check **Deploy Logs** in Railway dashboard |
| BERT model not found | Confirm `model.safetensors` was pushed via Git LFS |
| CORS error in browser | Verify Railway URL in `API_BASE` has no trailing slash |
| Supabase timeout | Check your Supabase project is not paused (free tier pauses after inactivity) |
| Railway out of memory | Upgrade to Hobby plan ($5/mo) — BERT needs ~1.5GB RAM |

---

## 💰 Cost Summary

| Service | Free Tier | Paid |
|---|---|---|
| **Railway** | 500MB RAM, limited hours | $5/mo Hobby |
| **Vercel** | Unlimited static hosting | Free |
| **Supabase** | 500MB DB, 2GB bandwidth | $25/mo Pro |
| **Groq** | Free tier (limited RPM) | Pay-per-use |
| **Git LFS** | 1GB free | $5/mo per 50GB |

> 💡 For a student project / demo: everything fits in free tiers.  
> Railway free tier may sleep after inactivity (cold start ~30s on first request).
