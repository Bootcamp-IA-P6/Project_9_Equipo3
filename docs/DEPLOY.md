# Deploy — Render (free tier)

This guide walks through deploying `youtube_hate_detector` to [Render](https://render.com)
using the Blueprint (`render.yaml`) at the root of the repo. The Blueprint provisions
two services: a Dockerised FastAPI backend (`signalmod-api`) and a static React/Vite
frontend (`signalmod-ui`).

## Prerequisites

- A free [Render](https://dashboard.render.com/register) account.
- This repository pushed to GitHub (Render reads the Blueprint from the connected branch).
- Credentials for any external services you plan to use:
  - `YOUTUBE_API_KEY` — Google Cloud Console → APIs & Services → Credentials.
  - `SUPABASE_URL` and `SUPABASE_KEY` — Supabase project settings (optional, only if
    persistence is enabled).

## One-shot deploy (Blueprint)

1. Open the Render dashboard → **New** → **Blueprint**.
2. Connect the GitHub account and pick this repository / branch.
3. Render parses `render.yaml` and shows the two services. Click **Apply**.
4. Wait for the first build. The API image takes ~5–8 minutes (frontend build +
   Python deps + spaCy/NLTK data).
5. Once the API is live, open **signalmod-api → Environment** and fill in the
   secrets that were declared with `sync: false`:
   - `YOUTUBE_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
6. Trigger a manual redeploy so the API picks up the new env vars.

The static site (`signalmod-ui`) reads `VITE_API_BASE_URL` at build time and points
at `https://signalmod-api.onrender.com`. If you renamed the API service, update that
env var in `render.yaml` and re-apply the Blueprint.

## Important free-tier caveats

- **Cold starts.** Free web services sleep after 15 minutes of inactivity. The first
  request after a sleep can take 30–60 seconds while the container restarts and the
  model warms up. Subsequent requests are fast.
- **Memory ceiling: 512 MB.** The default production model
  (`Meta-Feature Stacking (Production)`) loads BERT and needs roughly 700 MB of RAM,
  so it will OOM on free. The Blueprint pins `MODEL_NAME="LR + TF-IDF (Baseline)"`,
  which runs comfortably under the limit. If you upgrade to a paid plan with
  ≥1 GB RAM, you can switch the env var to the meta-stacking model and rebuild the
  image with `INSTALL_HF=1`.
- **Build image size.** `render.yaml` builds with the default `INSTALL_HF=0` so the
  Docker image stays small (no torch/transformers). Locally, `docker-compose.yml`
  still passes `INSTALL_HF=1` for parity with the production model.

## Test the Docker image locally before pushing

```bash
# Build the same image Render will build (free-tier shape — no HF extras)
docker build --build-arg INSTALL_HF=0 -t signalmod-api:render .

# Run it with the same model the Blueprint will use
docker run --rm -p 8000:8000 \
  -e MODEL_NAME="LR + TF-IDF (Baseline)" \
  -e ENV=production \
  signalmod-api:render

# Health check
curl http://localhost:8000/health
```

If `/health` returns `200 OK` locally, the same image will boot on Render.
