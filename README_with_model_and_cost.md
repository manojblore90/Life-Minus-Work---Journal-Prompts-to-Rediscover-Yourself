# Life Minus Work — Reflection Quiz

A Streamlit app that runs a 15‑question self‑reflection quiz and generates a rich, personalized PDF.
It supports free‑text answers, deep AI insights, score bars, a logo header, **Balancing Opportunity**
for your lowest themes, and a **Tiny Progress Tracker** at the end of the report.

---

## Folder layout

```
repo-root/
├─ main/
│  ├─ app.py                     # the Streamlit app
│  ├─ questions.json             # 15 questions + weights
│  ├─ Life-Minus-Work-Logo.webp  # optional; or logo.png
│  └─ logo.png                   # optional alternative
├─ requirements.txt
└─ README.md
```

> Put **questions.json** and your **logo file** in the **`main/`** folder, right next to `app.py`.

---

## 1) Running locally

### Prereqs
- Python **3.10+** recommended

### Install & run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run main/app.py
```

### Add your OpenAI key (optional, for AI insights)
Create an environment variable before you run:

**macOS / Linux**
```bash
export OPENAI_API_KEY="sk-...your-key..."
```

**Windows (PowerShell)**
```powershell
$env:OPENAI_API_KEY="sk-...your-key..."
```

If no key is set, the app falls back to a simple non‑AI narrative.

---

## 2) Deploying on Streamlit Cloud

1. Push your repo to GitHub with this structure (see above).
2. In Streamlit Cloud: **New app** → pick your repo, set **Main file** to `main/app.py`.
3. Go to **Settings → Secrets** and add:
   ```toml
   OPENAI_API_KEY = "sk-...your-key..."
   ```
4. Click **Rerun**.

---

## 3) Model & cost control

This project uses OpenAI’s **Responses API** and defaults to a fast/affordable model.

### Swap models quickly
Edit the line near the top of `main/app.py` **or** set an environment variable:

**Recommended (no code edit):**
```bash
# macOS / Linux
export OPENAI_MODEL="gpt-4o-mini"     # or gpt-5-nano
# Windows (PowerShell)
$env:OPENAI_MODEL="gpt-4o-mini"
```

**Then** change the line in `app.py` to read from the env var (one‑time change):
```python
# Before:
OPENAI_MODEL = "gpt-5-nano"
# After:
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
```

Now you can switch models per environment (local vs Cloud) without changing code.

### Control output length & style
Inside `ai_sections_and_weights(...)` in `app.py`, tune:
```python
temperature=0.7,          # lower = more deterministic, cheaper retries
max_output_tokens=900,    # reduce to ~450–600 to save tokens
```
Smaller prompts help too. The app already truncates free‑text answers to ~280 chars each.

### Toggle AI on/off
- **On:** set `OPENAI_API_KEY` (in Secrets or your shell env).
- **Off:** remove the key; the app automatically switches to the non‑AI fallback.
  (Optional: add a flag, e.g., `ENABLE_AI=0`, and gate on it in code if you want a hard switch.)

### Guardrails
- On any error (network, rate limit, etc.), the code **falls back** to a helpful non‑AI write‑up.
- Consider adding simple request metering if you expect heavy traffic (e.g., one AI call per minute per IP/user).

---

## 4) Logo

- Place `Life-Minus-Work-Logo.webp` or `logo.png` in `main/`.
- If `.webp` is present, the app converts it to `.png` internally for the PDF.
- If conversion fails, the app will simply omit the logo (the PDF still works).

---

## 5) Troubleshooting

**Blank page / “Could not find questions.json”**  
Make sure `questions.json` sits in the **same folder** as `app.py` (`main/`).

**“AI disabled (fallback used)”**  
Add your key in **Settings → Secrets** and rerun; or set the `OPENAI_API_KEY` env var locally.

**Unicode / weird characters in PDF**  
The app normalizes “curly” punctuation to ASCII so the built‑in PDF font won’t crash.
If you want full Unicode, we can bundle a TTF and switch to `add_font(..., uni=True)`.

**Logo not showing**  
Ensure the file path is `main/Life-Minus-Work-Logo.webp` or `main/logo.png`. WebP conversion needs Pillow.

---

## 6) Editing the questions

`main/questions.json` contains 15 questions, each with 4 choices and theme weights across:
`Identity, Growth, Connection, Peace, Adventure, Contribution`.

The app also offers a “✍️ I’ll write my own answer” option for every question and (if AI is enabled)
lets the model nudge scores based on what the user writes.

---

## Privacy & safety

- PDF is generated on the fly and offered as a download.
- A lightweight CSV is saved to `/tmp/responses.csv` in Cloud, which is **ephemeral**.
- No medical or clinical advice is provided; this is for reflection only.
