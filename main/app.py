import os, re, json, unicodedata, datetime
from pathlib import Path
import streamlit as st
from fpdf import FPDF
from PIL import Image
from io import BytesIO
from openai import OpenAI

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Life Minus Work – Reflection Quiz", layout="centered")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_COMPLETION_TOKENS = 8000  # deluxe cap

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def clean_text(s: str, max_len: int = 1000) -> str:
    """Bulletproof text cleaner for PDF output."""
    if not s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    # Normalize
    s = unicodedata.normalize("NFKC", s)
    # Remove control chars except newlines
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    # Collapse long tokens (safety)
    tokens = s.split()
    safe_tokens = []
    for t in tokens:
        if len(t) > max_len:
            safe_tokens.append(t[:max_len] + "…")
        else:
            safe_tokens.append(t)
    s = " ".join(safe_tokens)
    return s

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
# LOAD QUESTIONS
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

questions, _ = load_questions("questions.json")

# -----------------------------------------------------------------------------
# AI CALL
# -----------------------------------------------------------------------------
def ai_sections_and_weights(first_name, answers, scores, top3, lowest2, words):
    """Call OpenAI for structured reflection output."""
    prompt = f"""
Generate a JSON with deep, human insight for {first_name}.
Scores: {scores}
Top 3 themes: {top3}
Lowest 2 themes: {lowest2}
User's own words: {words}

JSON keys required:
archetype, core_need, deep_insight, why_now, strengths, energizers,
drainers, tensions, blindspot, actions, if_then, weekly_plan, affirmation,
quote, signature_metaphor, signature_sentence, top_theme_boosters, pitfalls,
future_snapshot, micro_pledge
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        usage = resp.usage

        data = json.loads(raw)

        safe_get = lambda key, default="": str(data.get(key, default) or "")
        out = {
            "archetype": safe_get("archetype"),
            "core_need": safe_get("core_need"),
            "deep_insight": safe_get("deep_insight"),
            "why_now": safe_get("why_now"),
            "strengths": [str(x) for x in (data.get("strengths") or [])][:6],
            "energizers": [str(x) for x in (data.get("energizers") or [])][:4],
            "drainers": [str(x) for x in (data.get("drainers") or [])][:4],
            "tensions": [str(x) for x in (data.get("tensions") or [])][:3],
            "blindspot": safe_get("blindspot"),
            "actions": [str(x) for x in (data.get("actions") or [])][:3],
            "if_then": [str(x) for x in (data.get("if_then") or [])][:3],
            "weekly_plan": [str(x) for x in (data.get("weekly_plan") or [])][:7],
            "affirmation": safe_get("affirmation"),
            "quote": safe_get("quote"),
            "signature_metaphor": safe_get("signature_metaphor"),
            "signature_sentence": safe_get("signature_sentence"),
            "top_theme_boosters": [str(x) for x in (data.get("top_theme_boosters") or [])][:4],
            "pitfalls": [str(x) for x in (data.get("pitfalls") or [])][:4],
            "future_snapshot": safe_get("future_snapshot"),
            "micro_pledge": safe_get("micro_pledge"),
        }

        st.caption(f"Token usage: input={usage.prompt_tokens}, output={usage.completion_tokens}, total={usage.total_tokens}")
        return out

    except Exception as e:
        st.error(f"AI generation failed: {e}")
        return {"deep_insight": "AI unavailable — using template."}

# -----------------------------------------------------------------------------
# PDF MAKER
# -----------------------------------------------------------------------------
def make_pdf_bytes(first_name, email, scores, top3, lowest2, ai_out, logo_path=None):
    pdf = FPDF()
    pdf.add_page()

    # Add logo
    if logo_path and Path(logo_path).exists():
        try:
            pdf.image(str(logo_path), 10, 8, 33)
        except:
            pass
    pdf.set_font("Helvetica", "B", 18)
    mc(pdf, f"{first_name}, your Reflection Report")

    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    mc(pdf, f"Email: {email}")
    mc(pdf, f"Top themes: {', '.join(top3)}")
    mc(pdf, f"Lowest themes: {', '.join(lowest2)}")

    pdf.ln(10)
    mc(pdf, "Your personalized AI reflection:")
    for k, v in ai_out.items():
        if v:
            pdf.set_font("Helvetica", "B", 12)
            mc(pdf, k.upper())
            pdf.set_font("Helvetica", "", 11)
            if isinstance(v, list):
                for item in v:
                    mc(pdf, f"- {item}")
            else:
                mc(pdf, v)
            pdf.ln(5)

    # Signature Week checklist page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    mc(pdf, "Signature Week – At a Glance Checklist")
    checklist = [
        "☐ Take one small step in your top theme",
        "☐ Share your reflection with a trusted friend",
        "☐ Revisit your affirmation each morning",
    ]
    for c in checklist:
        mc(pdf, c)

    return pdf.output(dest="S").encode("latin-1")

# -----------------------------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------------------------
st.title("Life Minus Work – Reflection Quiz")

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "first_name" not in st.session_state:
    st.session_state.first_name = ""

# Ask name first
st.session_state.first_name = st.text_input("Your first name:", st.session_state.first_name)

# Render questions
answers = {}
for i, q in enumerate(questions, 1):
    st.subheader(f"Q{i}. {q['question']}")
    choice = st.radio("Choose one:", q["options"], key=f"q{i}")
    free = st.text_input("Or write your own:", key=f"q{i}_free")
    answers[f"q{i}"] = free if free else choice

# After questions
if st.button("Generate my personalized report"):
    scores = {"Connection": 7, "Growth": 6, "Adventure": 5, "Identity": 4}  # dummy scoring
    top3 = ["Connection", "Growth", "Adventure"]
    lowest2 = ["Identity", "Health"]
    words = [v for v in answers.values() if v]

    ai_out = ai_sections_and_weights(
        st.session_state.first_name, answers, scores, top3, lowest2, words
    )

    logo_path = Path(__file__).parent / "Life-Minus-Work-Logo.webp"
    pdf_bytes = make_pdf_bytes(
        st.session_state.first_name,
        st.session_state.get("email", "not provided"),
        scores,
        top3,
        lowest2,
        ai_out,
        logo_path,
    )

    st.download_button("📥 Download your PDF report", pdf_bytes, file_name="Reflection_Report.pdf")
