import os, sys, re, json, unicodedata, datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import streamlit as st
from fpdf import FPDF
from PIL import Image
from openai import OpenAI

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
APP_TITLE = "Life Minus Work — Reflection Quiz (15 questions)"
REPORT_TITLE = "Your Reflection Report"
THEMES = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="centered")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
USE_AI = bool(OPENAI_API_KEY)
HIGH_MODEL = "gpt-5-mini"
MAX_TOK_HIGH = 8000
FALLBACK_CAP = 6000

# -----------------------------------------------------------------------------
# CLEANING + PDF SAFETY
# -----------------------------------------------------------------------------
def clean_text(s: str, max_len: int = 1000) -> str:
    """Bulletproof text cleaner for PDF output."""
    if not s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)  # strip bad controls
    tokens = s.split()
    safe_tokens = []
    for t in tokens:
        if len(t) > max_len:
            safe_tokens.append(t[:max_len] + "…")
        else:
            safe_tokens.append(t)
    return " ".join(safe_tokens)

def mc(pdf: "FPDF", text: str, h: float = 6):
    """Safe MultiCell that guarantees no crash, ever."""
    s = clean_text(text or "")
    try:
        pdf.multi_cell(0, h, s)
    except Exception:
        try:
            for line in s.split("\n"):
                pdf.multi_cell(0, h, clean_text(line))
        except Exception:
            pdf.multi_cell(0, h, "[…content truncated…]")

# -----------------------------------------------------------------------------
# QUESTIONS LOADER
# -----------------------------------------------------------------------------
def load_questions(filename="questions.json"):
    base_dir = Path(__file__).parent
    path = base_dir / filename
    if not path.exists():
        st.error(f"Could not find {filename} at {path}")
        st.stop()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"], data.get("themes", [])

# -----------------------------------------------------------------------------
# SCORING
# -----------------------------------------------------------------------------
def compute_scores(answers: dict, questions: list) -> Dict[str, int]:
    scores = {t: 0 for t in THEMES}
    for q in questions:
        qid = q["id"]
        choice_idx = answers.get(qid, {}).get("choice_idx")
        if choice_idx is None: continue
        try:
            choice = q["choices"][choice_idx]
        except (IndexError, KeyError, TypeError):
            continue
        for theme, val in choice.get("weights", {}).items():
            scores[theme] = scores.get(theme, 0) + val
    return scores

def top_themes(scores: Dict[str, int], k: int = 3) -> List[str]:
    return [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

def balancing_suggestion(theme: str) -> str:
    suggestions = {
        "Identity": "Choose one tiny ritual that reflects who you are becoming.",
        "Growth": "Pick a single skill and block 15 minutes to practice today.",
        "Connection": "Send a 3-line check-in to someone who matters.",
        "Peace": "Name a 10-minute wind-down you will repeat daily.",
        "Adventure": "Plan a 30–60 minute micro-adventure within 7 days.",
        "Contribution": "Offer one concrete act of help this week.",
    }
    return suggestions.get(theme, "Take one small, visible step this week.")

# -----------------------------------------------------------------------------
# OPENAI CALL
# -----------------------------------------------------------------------------
def _call_openai_json(model: str, system: str, user: str, cap: int):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=cap,
        response_format={"type": "json_object"},
    )
    return r.choices[0].message.content, r.usage

def ai_sections_and_weights(scores, top3, free_responses, first_name, horizon_weeks=4):
    if not USE_AI: return None
    try:
        packed = [{"id": fr.get("id", f"free_{i+1}"),
                   "q": clean_text(fr.get("question","")),
                   "a": clean_text(fr.get("answer",""))}
                  for i, fr in enumerate(free_responses) if fr.get("answer")]
        score_lines = ", ".join([f"{k}:{v}" for k,v in scores.items()])
        prompt = f"""
You are a life coach. Return STRICT JSON.
Keys: archetype, core_need, deep_insight, why_now, strengths, energizers,
drainers, tensions, blindspot, actions, if_then, weekly_plan, affirmation,
quote, signature_metaphor, signature_sentence, top_theme_boosters, pitfalls,
future_snapshot, micro_pledge.
User: {first_name}
Scores: {score_lines}
Top themes: {', '.join(top3)}
Free responses: {json.dumps(packed, ensure_ascii=False)}
"""
        raw, usage = _call_openai_json(HIGH_MODEL, "Return JSON only", prompt, MAX_TOK_HIGH)
        data = json.loads(raw)

        safe_get = lambda k: str(data.get(k, "") or "")
        out = {
            "archetype": safe_get("archetype"),
            "core_need": safe_get("core_need"),
            "deep_insight": safe_get("deep_insight"),
            "why_now": safe_get("why_now"),
            "strengths": [str(x) for x in data.get("strengths", [])][:6],
            "energizers": [str(x) for x in data.get("energizers", [])][:4],
            "drainers": [str(x) for x in data.get("drainers", [])][:4],
            "tensions": [str(x) for x in data.get("tensions", [])][:3],
            "blindspot": safe_get("blindspot"),
            "actions": [str(x) for x in data.get("actions", [])][:3],
            "if_then": [str(x) for x in data.get("if_then", [])][:3],
            "weekly_plan": [str(x) for x in data.get("weekly_plan", [])][:7],
            "affirmation": safe_get("affirmation"),
            "quote": safe_get("quote"),
            "signature_metaphor": safe_get("signature_metaphor"),
            "signature_sentence": safe_get("signature_sentence"),
            "top_theme_boosters": [str(x) for x in data.get("top_theme_boosters", [])][:4],
            "pitfalls": [str(x) for x in data.get("pitfalls", [])][:4],
            "future_snapshot": safe_get("future_snapshot"),
            "micro_pledge": safe_get("micro_pledge"),
        }
        st.caption(f"AI tokens: in={usage.prompt_tokens}, out={usage.completion_tokens}, total={usage.total_tokens}")
        return out
    except Exception as e:
        st.error(f"AI error: {e}")
        return {"deep_insight": "AI unavailable"}

# -----------------------------------------------------------------------------
# PDF
# -----------------------------------------------------------------------------
def make_pdf_bytes(first_name, email, scores, top3, sections, free_responses):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    mc(pdf, f"{first_name}, your Reflection Report")

    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    mc(pdf, f"Email: {email}")
    mc(pdf, f"Top themes: {', '.join(top3)}")

    pdf.ln(10)
    for k,v in sections.items():
        if not v: continue
        pdf.set_font("Helvetica", "B", 12); mc(pdf, k.upper())
        pdf.set_font("Helvetica", "", 11)
        if isinstance(v, list):
            for item in v: mc(pdf, f"- {item}")
        else:
            mc(pdf, v)
        pdf.ln(5)

    # Checklist page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    mc(pdf, "Signature Week – At a Glance")
    for c in ["☐ One tiny step", "☐ Share reflection", "☐ Review weekly"]:
        mc(pdf, c)

    return pdf.output(dest="S").encode("latin-1")

# -----------------------------------------------------------------------------
# APP UI
# -----------------------------------------------------------------------------
st.title(APP_TITLE)
first_name = st.text_input("First name", "")
questions, _ = load_questions()
answers = {}
free_responses = []

for q in questions:
    st.subheader(q["text"])
    opts = [c["label"] for c in q["choices"]] + ["✍️ My own answer"]
    sel = st.radio("Choose one:", opts, index=None, key=q["id"])
    free_text = ""
    if sel == "✍️ My own answer":
        free_text = st.text_area("Your answer", key=f"{q['id']}_free")
    else:
        idx = opts.index(sel) if sel else None
        answers[q["id"]] = {"choice_idx": idx}
    if free_text:
        free_responses.append({"id": q["id"], "question": q["text"], "answer": free_text})

email = st.text_input("Email for report")

if st.button("Generate Report"):
    scores = compute_scores(answers, questions)
    top3 = top_themes(scores, 3)
    sections = ai_sections_and_weights(scores, top3, free_responses, first_name)
    pdf_bytes = make_pdf_bytes(first_name, email, scores, top3, sections or {}, free_responses)
    st.download_button("📥 Download PDF", pdf_bytes, "Reflection_Report.pdf")
