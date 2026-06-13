# Deployment guide

This is a **FastAPI Python server** with an in-memory vector store. It needs a
host that runs a long-lived web process. **Netlify (static/serverless) is not a
good fit** — use **Render** (easiest free option), **Railway**, **Fly.io**, or
**Hugging Face Spaces** (Docker).

The repo already includes everything needed:

| File             | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| `render.yaml`    | One-click config for Render                         |
| `Procfile`       | Start command for Railway / Heroku-style hosts      |
| `runtime.txt`    | Pins Python 3.12                                     |
| `Dockerfile`     | For Hugging Face Spaces / Fly.io / any Docker host  |
| `requirements.txt` | Includes `google-genai` for Gemini                |

---

## Important environment variables

| Variable             | Value                  | Why                                              |
| -------------------- | ---------------------- | ------------------------------------------------ |
| `GEMINI_API_KEY`     | *(your key)*           | Enables Gemini embeddings + generation           |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Embedding model                                  |
| `GEMINI_CHAT_MODEL`  | `gemini-2.5-flash`     | Chat model                                       |
| `AUTO_INDEX_PATH`    | `backend`              | Auto-index bundled code on startup (demo content)|
| `INDEX_BASE_DIR`     | `.`                    | Security: limits `/index` to the project folder  |

> **Never commit your key.** It is read from the host's environment variables.
> `.env` is git-ignored.

---

## Step 1 — Push the project to GitHub

```powershell
cd c:\Personal\projects\ai-code-assistant
git init
git add .
git commit -m "AI code assistant - deploy ready"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-code-assistant.git
git push -u origin main
```

Confirm `.env` is **not** in the commit (it is git-ignored).

---

## Step 2 — Deploy on Render (recommended)

1. Go to <https://render.com> → sign in with GitHub.
2. **New +** → **Web Service** → pick your `ai-code-assistant` repo.
3. Render auto-detects `render.yaml`. Confirm:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment**, add `GEMINI_API_KEY` = *your key* (the other vars come
   from `render.yaml`).
5. Click **Create Web Service**. First build takes a few minutes.
6. Open the generated URL, e.g. `https://ai-code-assistant.onrender.com` — the
   UI loads and the backend code is already indexed.

> Free tier sleeps after inactivity; the first request after idle is slow. The
> in-memory index is rebuilt on each cold start via `AUTO_INDEX_PATH`.

---

## Alternative — Railway

1. <https://railway.app> → **New Project** → **Deploy from GitHub repo**.
2. Railway uses the `Procfile` automatically.
3. Add the same environment variables (Variables tab).
4. Generate a public domain under **Settings → Networking**.

## Alternative — Hugging Face Spaces (Docker)

1. Create a new **Space** → SDK: **Docker**.
2. Push this repo (it uses the included `Dockerfile`).
3. Add `GEMINI_API_KEY` under **Settings → Variables and secrets**.

---

## (Optional) Frontend on Netlify

If you specifically want Netlify, only the **frontend** can go there; the
backend still needs Render/Railway:

1. Deploy the backend first (Step 2) and note its URL.
2. In `frontend/index.html`, change the `fetch('/health')`, `fetch('/index')`,
   `fetch('/ask')`, and `fetch('/ask/stream')` calls to use the full backend
   URL (e.g. `https://ai-code-assistant.onrender.com/ask`).
3. On Netlify: **Add new site → Deploy manually**, drag the `frontend` folder.
4. Enable CORS on the backend (add `fastapi.middleware.cors.CORSMiddleware`)
   so the Netlify origin can call it.

> Simpler: skip Netlify — Render already serves the UI at `/`.

---

## Verify a deployment

```powershell
curl https://<your-app>.onrender.com/health
```

Expected: `"gemini_enabled": true` and a non-zero `indexed_chunks`.
