
# Life Minus Work — Streamlit (Full 15Q) — `main/` folder layout

This package is structured for Streamlit Cloud where **Main file path = `main/app.py`**.

Repo root contains:
- `runtime.txt`  → pins Python 3.10
- `requirements.txt`
- `README.md`
- `main/` folder with:
  - `app.py` (full app with diagnostics, PDF, /tmp CSV, optional AI)
  - `questions.json` (full 15 questions)

## Deploy steps
1) Upload these files to the **root of your GitHub repo**. You should see `runtime.txt` and a `main/` folder.
2) In Streamlit Cloud → App → **Settings → Main file path** = `main/app.py`
3) Click **Rerun** (or redeploy).
4) (Optional) Add `OPENAI_API_KEY` in **Settings → Secrets** to enable AI narrative.

If the page is blank, the diagnostic sidebar and header in `app.py` will still render so we can see what's happening.
