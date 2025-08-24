
import os
import json
import datetime
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import streamlit as st
from fpdf import FPDF
from PIL import Image

APP_TITLE = "Life Minus Work — Reflection Quiz (15 questions)"
REPORT_TITLE = "Your Reflection Report"
THEMES = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

# Read model from env if provided, else default
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")  # or "gpt-4o-mini"

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="centered")
st.title(APP_TITLE)
st.write("Answer 15 questions, add your own reflections, and instantly download a personalized PDF summary.")

# ---------- Data loading ----------
def load_questions(filename: str = "questions.json") -> Tuple[List[dict], List[str]]:
    base_dir = Path(__file__).parent
    path = base_dir / filename
    if not path.exists():
        st.error(f"Could not find {filename} at {path}. It must sit next to main/app.py.")
        st.stop()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"], data.get("themes", [])

# ---------- Logo loader (webp -> png if needed) ----------
def get_logo_png_path() -> Optional[str]:
    """Try several common locations. If WEBP found, convert to /tmp/logo.png."""
    here = Path(__file__).parent
    candidates = [
        here / "logo.png",
        here / "Life-Minus-Work-Logo.png",
        here / "Life-Minus-Work-Logo.webp",
        here / "assets" / "logo.png",
        here / "assets" / "Life-Minus-Work-Logo.png",
        here / "assets" / "Life-Minus-Work-Logo.webp",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".png":
                return str(p)
            if p.suffix.lower() == ".webp":
                try:
                    img = Image.open(p).convert("RGBA")
                    out = Path("/tmp/logo.png")
                    img.save(out, format="PNG")
                    return str(out)
                except Exception:
                    return None
    return None

# ---------- Fonts (optional Unicode via TTF) ----------
def try_add_unicode_font(pdf: FPDF) -> bool:
    """If main/fonts/DejaVuSans.ttf exists, register it and use it for full Unicode."""
    here = Path(__file__).parent
    font_path = here / "fonts" / "DejaVuSans.ttf"
    if font_path.exists():
        try:
            pdf.add_font("DejaVu", "", str(font_path), uni=True)
            pdf.add_font("DejaVu", "B", str(font_path), uni=True)
            return True
        except Exception:
            return False
    return False

# ---------- PDF text safety (fallback for core fonts) ----------
def safe_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = (s.replace("\u2019", "'")   # ’
           .replace("\u2018", "'")  # ‘
           .replace("\u201c", '"') # “
           .replace("\u201d", '"') # ”
           .replace("\u2013", "-")  # –
           .replace("\u2014", "-")) # —
    s = unicodedata.normalize("NFKD", s).encode("latin-1", "ignore").decode("latin-1")
    return s

# ---------- Optional AI ----------
USE_AI = bool(os.getenv("OPENAI_API_KEY"))

def ai_sections_and_weights(
    scores: Dict[str, int],
    top3: List[str],
    free_responses: List[dict],
    first_name: str
) -> Optional[dict]:
    """Deep read + weights for free-text answers."""
    if not USE_AI:
        return None
    try:
        from openai import OpenAI
        client = OpenAI()

        score_lines = ", ".join([f"{k}: {v}" for k, v in scores.items()])
        packed = [
            {"id": fr["id"], "q": fr["question"], "a": str(fr.get("answer", ""))[:280]}
            for fr in free_responses if fr.get("answer")
        ]

        prompt = (
            "You are a warm, practical life coach. Return ONLY valid JSON with keys:\n"
            "  archetype (string), core_need (string),\n"
            "  deep_insight (string, 140-220 words, address the user by first name),\n"
            "  why_now (string, 60-100 words),\n"
            "  strengths (array of 3-5 short strings),\n"
            "  energizers (array of 3), drainers (array of 3),\n"
            "  tensions (array of 1-2 short strings), blindspot (string <= 40 words),\n"
            "  actions (array of EXACTLY 3 short bullet strings),\n"
            "  if_then (array of EXACTLY 3 implementation-intention strings like: 'If it’s 7pm, then I…'),\n"
            "  weekly_plan (array of 7 brief day-plan strings),\n"
            "  affirmation (string <= 15 words), quote (string <= 20 words),\n"
            "  signature_metaphor (string <= 12 words), signature_sentence (string <= 20 words),\n"
            "  top_theme_boosters (array of up to 3 short suggestions), pitfalls (array of up to 3),\n"
            "  weights (object mapping question_id -> object of theme:int in [-2,2]).\n"
            f"User first name: {first_name or 'Friend'}.\n"
            f"Theme scores so far: {score_lines}.\n"
            f"Top 3 themes: {', '.join(top3)}.\n"
            "Also consider these free-text answers (omit weights for questions you don't see):\n"
            f"{json.dumps(packed, ensure_ascii=False)}\n"
            "Tone: empathetic, encouraging, plain language. No medical/clinical claims. JSON only."
        )

        resp = client.responses.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": "Reply with helpful coaching guidance as STRICT JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_output_tokens=1100,
        )
        raw = resp.output_text or "{}"
        data = json.loads(raw)

        out = {
            "archetype": str(data.get("archetype", "")),
            "core_need": str(data.get("core_need", "")),
            "deep_insight": str(data.get("deep_insight", "")),
            "why_now": str(data.get("why_now", "")),
            "strengths": [str(x) for x in (data.get("strengths") or [])][:5],
            "energizers": [str(x) for x in (data.get("energizers") or [])][:3],
            "drainers": [str(x) for x in (data.get("drainers") or [])][:3],
            "tensions": [str(x) for x in (data.get("tensions") or [])][:2],
            "blindspot": str(data.get("blindspot", "")),
            "actions": [str(x) for x in (data.get("actions") or [])][:3],
            "if_then": [str(x) for x in (data.get("if_then") or [])][:3],
            "weekly_plan": [str(x) for x in (data.get("weekly_plan") or [])][:7],
            "affirmation": str(data.get("affirmation", "")),
            "quote": str(data.get("quote", "")),
            "signature_metaphor": str(data.get("signature_metaphor", "")),
            "signature_sentence": str(data.get("signature_sentence", "")),
            "top_theme_boosters": [str(x) for x in (data.get("top_theme_boosters") or [])][:3],
            "pitfalls": [str(x) for x in (data.get("pitfalls") or [])][:3],
            "weights": {},
        }

        weights = data.get("weights") or {}
        for qid, w in weights.items():
            clean = {}
            if isinstance(w, dict):
                for theme, val in w.items():
                    if theme in THEMES:
                        try:
                            iv = int(val)
                            iv = max(-2, min(2, iv))
                            clean[theme] = iv
                        except Exception:
                            pass
            if clean:
                out["weights"][qid] = clean
        return out
    except Exception:
        return None

# ---------- Scoring ----------
def compute_scores(answers: dict, questions: list) -> Dict[str, int]:
    scores = {t: 0 for t in THEMES}
    for q in questions:
        qid = q["id"]
        choice_idx = answers.get(qid, {}).get("choice_idx")
        if choice_idx is None:
            continue
        try:
            choice = q["choices"][choice_idx]
        except (IndexError, KeyError, TypeError):
            continue
        for theme, val in choice.get("weights", {}).items():
            scores[theme] = scores.get(theme, 0) + val
    return scores

def apply_free_text_weights(scores: Dict[str, int], ai_weights: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    for qid, wmap in ai_weights.items():
        for theme, delta in wmap.items():
            scores[theme] = scores.get(theme, 0) + int(delta)
    return scores

def top_themes(scores: Dict[str, int], k: int = 3) -> List[str]:
    return [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

def lowest_themes(scores: Dict[str, int], k: int = 2) -> List[str]:
    return [name for name, _ in sorted(scores.items(), key=lambda x: x[1])[:k]]

def balancing_suggestion(theme: str) -> str:
    suggestions = {
        "Identity": "Choose one tiny ritual that reflects who you’re becoming.",
        "Growth": "Pick a single skill and block 15 minutes to practice today.",
        "Connection": "Send a 3-line check-in to someone who matters.",
        "Peace": "Name a 10-minute wind-down you’ll repeat daily.",
        "Adventure": "Plan a 30–60 minute micro-adventure within 7 days.",
        "Contribution": "Offer one concrete act of help this week.",
    }
    return suggestions.get(theme, "Take one small, visible step this week.")

# ---------- Pretty PDF ----------
def draw_scores_barchart(pdf: FPDF, scores: Dict[str, int], use_unicode: bool):
    pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14)
    pdf.cell(0, 8, ("Your Theme Snapshot"), ln=True)
    pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
    max_score = max(max(scores.values()), 1)
    bar_w_max = 120
    x_left = pdf.get_x() + 10
    y = pdf.get_y()
    for theme in THEMES:
        val = scores.get(theme, 0)
        bar_w = (val / max_score) * bar_w_max
        pdf.set_xy(x_left, y)
        pdf.cell(35, 6, theme if use_unicode else theme)
        pdf.set_fill_color(30, 144, 255)
        pdf.rect(x_left + 38, y + 1.3, bar_w, 4.5, "F")
        pdf.set_xy(x_left + 38 + bar_w + 2, y)
        pdf.cell(0, 6, str(val))
        y += 7
    pdf.set_y(y + 4)

def paragraph(pdf: FPDF, title: str, body: str, use_unicode: bool):
    pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14)
    pdf.cell(0, 8, (title), ln=True)
    pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
    text = body if use_unicode else safe_text(body)
    for line in text.split("\n"):
        pdf.multi_cell(0, 6, line)
    pdf.ln(2)

def bullets(pdf: FPDF, title: str, items: List[str], use_unicode: bool):
    if not items: return
    pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14)
    pdf.cell(0, 8, (title), ln=True)
    pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
    for it in items:
        pdf.cell(4, 6, "*")
        pdf.multi_cell(0, 6, it if use_unicode else safe_text(it))
    pdf.ln(1)

def checkbox_line(pdf: FPDF, text: str, use_unicode: bool):
    x = pdf.get_x()
    y = pdf.get_y()
    pdf.rect(x, y + 1.5, 4, 4)  # little square
    pdf.set_x(x + 6)
    pdf.multi_cell(0, 6, text if use_unicode else safe_text(text))

def label_value(pdf: FPDF, label: str, value: str, use_unicode: bool):
    pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 12); pdf.cell(0, 6, (label), ln=True)
    pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12);  pdf.multi_cell(0, 6, value if use_unicode else safe_text(value))

def make_pdf_bytes(first_name: str, email: str, scores: Dict[str,int], top3: List[str],
                   sections: dict, free_responses: List[dict], logo_path: Optional[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    use_unicode = try_add_unicode_font(pdf)

    # Logo (optional)
    if logo_path:
        try:
            pdf.image(logo_path, w=40); pdf.ln(2)
        except Exception:
            pass

    # Title block
    pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 18)
    pdf.cell(0, 10, (REPORT_TITLE), ln=True)
    pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
    today = datetime.date.today().strftime("%d %b %Y")
    greet = f"Hi {first_name}," if first_name else "Hello,"
    pdf.cell(0, 8, (greet if use_unicode else safe_text(greet)), ln=True)
    pdf.cell(0, 8, (f"Date: {today}"), ln=True)
    if email:
        pdf.cell(0, 8, (f"Email: {email}"), ln=True)
    pdf.ln(1)

    # Archetype/core & signature
    if sections.get("archetype") or sections.get("core_need"):
        label_value(pdf, "Archetype", sections.get("archetype","") or "—", use_unicode)
        label_value(pdf, "Core Need", sections.get("core_need","") or "—", use_unicode)
        if sections.get("signature_metaphor"):
            label_value(pdf, "Signature Metaphor", sections.get("signature_metaphor",""), use_unicode)
        if sections.get("signature_sentence"):
            label_value(pdf, "Signature Sentence", sections.get("signature_sentence",""), use_unicode)
        pdf.ln(1)

    # Top themes
    pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14); pdf.cell(0, 8, ("Top Themes"), ln=True)
    pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12);  pdf.multi_cell(0, 6, (", ".join(top3) if use_unicode else safe_text(", ".join(top3)))); pdf.ln(1)

    # Score bars
    draw_scores_barchart(pdf, scores, use_unicode)

    # Insight blocks
    if sections.get("deep_insight"):
        paragraph(pdf, "What this really says about you", sections["deep_insight"], use_unicode)

    if sections.get("why_now"):
        label_value(pdf, "Why this matters now", sections["why_now"], use_unicode); pdf.ln(1)

    # Strengths / Energizers / Drainers
    if sections.get("strengths"):
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14); pdf.cell(0, 8, "Signature strengths", ln=True)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
        for s in sections["strengths"]:
            pdf.cell(4, 6, "*"); pdf.multi_cell(0, 6, (s if use_unicode else safe_text(s)))
        pdf.ln(1)

    if sections.get("energizers") or sections.get("drainers"):
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14); pdf.cell(0, 8, "Energy map", ln=True)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 12); pdf.cell(0, 6, "Energizers", ln=True)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
        for e in sections.get("energizers", []):
            pdf.cell(4, 6, "+"); pdf.multi_cell(0, 6, (e if use_unicode else safe_text(e)))
        pdf.ln(1)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 12); pdf.cell(0, 6, "Drainers", ln=True)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
        for d in sections.get("drainers", []):
            pdf.cell(4, 6, "–"); pdf.multi_cell(0, 6, (d if use_unicode else safe_text(d)))
        pdf.ln(1)

    # Tensions & blindspot
    if sections.get("tensions"):
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14); pdf.cell(0, 8, "Hidden tensions", ln=True)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
        for t in sections["tensions"]:
            pdf.cell(4, 6, "*"); pdf.multi_cell(0, 6, (t if use_unicode else safe_text(t)))
        pdf.ln(1)
    if sections.get("blindspot"):
        label_value(pdf, "Watch-out (gentle blind spot)", sections["blindspot"], use_unicode); pdf.ln(1)

    # Actions & If–Then plans
    if sections.get("actions"):
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14); pdf.cell(0, 8, "3 next-step actions (7 days)", ln=True)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
        for a in sections["actions"]:
            checkbox_line(pdf, a, use_unicode)
        pdf.ln(1)

    if sections.get("if_then"):
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14); pdf.cell(0, 8, "Implementation intentions (If–Then)", ln=True)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
        for it in sections["if_then"]:
            pdf.cell(4, 6, "*"); pdf.multi_cell(0, 6, (it if use_unicode else safe_text(it)))
        pdf.ln(1)

    # Weekly plan
    if sections.get("weekly_plan"):
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14); pdf.cell(0, 8, "1-week gentle plan", ln=True)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
        for i, item in enumerate(sections["weekly_plan"][:7]):
            pdf.cell(0, 6, (f"Day {i+1}: {item}" if use_unicode else safe_text(f"Day {i+1}: {item}")), ln=True)
        pdf.ln(1)

    # Balancing Opportunity (lowest 1–2 themes)
    lows = [name for name, _ in sorted(scores.items(), key=lambda x: x[1])[:2]]
    if lows:
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14); pdf.cell(0, 8, "Balancing Opportunity", ln=True)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
        for theme in lows:
            tip = balancing_suggestion(theme)
            pdf.multi_cell(0, 6, (f"{theme}: {tip}" if use_unicode else safe_text(f"{theme}: {tip}")))
        pdf.ln(1)

    # Boosters & Pitfalls (for top themes)
    if sections.get("top_theme_boosters") or sections.get("pitfalls"):
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14); pdf.cell(0, 8, "Amplify what works / Avoid what trips you", ln=True)
        if sections.get("top_theme_boosters"):
            pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 12); pdf.cell(0, 6, "Boosters", ln=True)
            pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
            for b in sections.get("top_theme_boosters", []):
                pdf.cell(4, 6, "*"); pdf.multi_cell(0, 6, (b if use_unicode else safe_text(b)))
        if sections.get("pitfalls"):
            pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 12); pdf.cell(0, 6, "Pitfalls", ln=True)
            pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
            for p in sections.get("pitfalls", []):
                pdf.cell(4, 6, "–"); pdf.multi_cell(0, 6, (p if use_unicode else safe_text(p)))
        pdf.ln(1)

    # Quote & affirmation
    if sections.get("affirmation") or sections.get("quote"):
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 12); pdf.cell(0, 6, "Keep this in view", ln=True)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "I", 11)
        if sections.get("affirmation"):
            pdf.multi_cell(0, 6, (f"Affirmation: {sections['affirmation']}" if use_unicode else safe_text(f"Affirmation: {sections['affirmation']}")))
        if sections.get("quote"):
            # smart quotes OK in Unicode font; else mapped via safe_text earlier
            qtext = f"“{sections['quote']}”" if use_unicode else f"\u201c{sections['quote']}\u201d"
            pdf.multi_cell(0, 6, (qtext if use_unicode else safe_text(qtext)))
        pdf.ln(2)

    # Your reflections (free text)
    if free_responses:
        pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14); pdf.cell(0, 8, "Your words we heard", ln=True)
        pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
        for fr in free_responses:
            if not fr.get("answer"): continue
            pdf.multi_cell(0, 6, (f"• {fr['question']}" if use_unicode else safe_text(f"• {fr['question']}")))
            pdf.multi_cell(0, 6, (f"  {fr['answer']}" if use_unicode else safe_text(f"  {fr['answer']}")))
            pdf.ln(1)

    # New page: Signature Week (at a glance)
    pdf.add_page()
    pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 16)
    pdf.cell(0, 10, ("Signature Week — At a glance"), ln=True)
    pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
    pdf.multi_cell(0, 6, ("A simple plan you can print or screenshot. Check items off as you go." if use_unicode else safe_text("A simple plan you can print or screenshot. Check items off as you go.")))
    pdf.ln(2)

    # Draw a simple 7-row checklist table for the weekly_plan (or fall back)
    week_items = sections.get("weekly_plan") or []
    if not week_items:
        week_items = [f"Do one small action for {t}" for t in top3] + ["Reflect and set next step"]

    for i, item in enumerate(week_items[:7]):
        checkbox_line(pdf, f"Day {i+1}: {item}", use_unicode)

    # Tiny Progress Tracker (checkboxes) — repeat here for visibility
    pdf.ln(2)
    pdf.set_font("DejaVu" if use_unicode else "Arial", "B", 14); pdf.cell(0, 8, "Tiny Progress Tracker", ln=True)
    pdf.set_font("DejaVu" if use_unicode else "Arial", "", 12)
    milestones = sections.get("actions") or [
        "Choose one tiny step and schedule it.",
        "Tell a friend your plan for gentle accountability.",
        "Spend 20 minutes on your step and celebrate completion."
    ]
    for m in milestones[:3]:
        checkbox_line(pdf, m, use_unicode)
    pdf.ln(2)

    # Footer
    pdf.set_font("DejaVu" if use_unicode else "Arial", "I", 10); pdf.ln(2)
    pdf.multi_cell(0, 5, ("Life Minus Work • This report is a starting point for reflection. Nothing here is medical or financial advice." if use_unicode else safe_text("Life Minus Work • This report is a starting point for reflection. Nothing here is medical or financial advice.")))
    return pdf.output(dest="S").encode("latin-1")

# ---------- UI ----------
# Step 1: First name
with st.form("intro_form"):
    first_name = st.text_input("First name", placeholder="e.g., Alex")
    started = st.form_submit_button("Start")
    if started and not first_name:
        st.error("Please enter your first name to continue.")
if "first_name" not in st.session_state:
    st.session_state["first_name"] = ""
if started and first_name:
    st.session_state["first_name"] = first_name.strip()

if st.session_state.get("first_name"):
    st.header("Your Questions")
    st.caption("Each question has choices *or* you can write your own answer.")

    questions, _ = load_questions("questions.json")
    # answers dict: qid -> {"choice_idx": int|None, "free_text": str|None}
    answers: Dict[str, dict] = {}
    free_responses: List[dict] = []

    for q in questions:
        st.subheader(q["text"])
        options = [c["label"] for c in q["choices"]] + ["✍️ I’ll write my own answer"]
        selected = st.radio("Choose one:", options, index=None, key=f"{q['id']}_choice")
        choice_idx = None
        free_text_val = None

        if selected == "✍️ I’ll write my own answer":
            free_text_val = st.text_area("Your answer", key=f"{q['id']}_free", height=80, placeholder="Type your own response...")
        elif selected is not None:
            choice_idx = options.index(selected)
            if choice_idx == len(options) - 1:  # the free-text option
                choice_idx = None

        answers[q["id"]] = {"choice_idx": choice_idx, "free_text": free_text_val}
        if free_text_val:
            free_responses.append({"id": q["id"], "question": q["text"], "answer": free_text_val})

        st.divider()

    # Step 3: Email & Download
    st.subheader("Email & Download")
    with st.form("finish_form"):
        email = st.text_input("Your email (for your download link)", placeholder="you@example.com")
        consent = st.checkbox("I agree to receive my results and occasional updates from Life Minus Work.")
        ready = st.form_submit_button("Generate My Personalized Report")

        if ready and (not email or not consent):
            st.error("Please enter your email and give consent to continue.")

    if ready and email and consent:
        # Compute scores from choices
        scores = compute_scores(answers, questions)

        # If AI available, ask for sections + weights for free answers
        sections = {"summary": "", "actions": [], "weekly_plan": [], "weights": {}}
        if USE_AI:
            maybe = ai_sections_and_weights(scores, top_themes(scores), free_responses, st.session_state.get("first_name",""))
            if maybe:
                sections.update(maybe)
                if sections.get("weights"):
                    scores = apply_free_text_weights(scores, sections["weights"])

        # Fallback sections if AI off or failed
        if not sections.get("deep_insight"):
            base = f"Thank you for completing the Reflection Quiz, {st.session_state.get('first_name','Friend')}."
            actions = [
                "Choose one tiny step you can take this week.",
                "Tell a friend your plan—gentle accountability.",
                "Schedule 20 minutes for reflection or journaling.",
            ]
            plan = [
                "Name your intention.",
                "15–20 minutes of learning or practice.",
                "Reach out to someone who energizes you.",
                "Take a calm walk or mindful pause.",
                "Do one small adventurous thing.",
                "Offer help or encouragement to someone.",
                "Review your week and set the next tiny step.",
            ]
            sections = {
                "deep_insight": base,
                "actions": actions,
                "weekly_plan": plan,
                "weights": {},
                "archetype": "",
                "core_need": "",
                "affirmation": "",
                "quote": "",
                "signature_metaphor": "",
                "signature_sentence": "",
                "top_theme_boosters": [],
                "pitfalls": [],
                "tensions": [],
                "blindspot": "",
            }

        top3 = top_themes(scores, 3)

        # PDF
        logo_path = get_logo_png_path()
        pdf_bytes = make_pdf_bytes(
            st.session_state.get("first_name",""),
            email,
            scores,
            top3,
            sections,
            free_responses,
            logo_path,
        )

        st.success("Your personalized report is ready!")
        st.download_button(
            "Download Your PDF Report",
            data=pdf_bytes,
            file_name="LifeMinusWork_Reflection_Report.pdf",
            mime="application/pdf",
        )

        # Save to /tmp (Cloud-safe, ephemeral)
        try:
            import csv
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            csv_path = "/tmp/responses.csv"
            file_exists = Path(csv_path).exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "first_name", "email", "scores", "top3"])
                writer.writerow([
                    ts,
                    st.session_state.get("first_name",""),
                    email,
                    json.dumps(scores),
                    json.dumps(top3),
                ])
            st.caption("Saved to /tmp/responses.csv (Cloud-safe, ephemeral).")
        except Exception as e:
            st.caption(f"Could not save responses (demo only). {e}")
else:
    st.info("Start by entering your first name above.")
