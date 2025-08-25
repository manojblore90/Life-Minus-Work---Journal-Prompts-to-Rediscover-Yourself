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

# Deluxe defaults (force High every time)
HIGH_MODEL = os.getenv("OPENAI_HIGH_MODEL", "gpt-5-mini")
MAX_TOK_HIGH = int(os.getenv("MAX_OUTPUT_TOKENS_HIGH", 15000))

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="centered")
st.title(APP_TITLE)
st.write("Answer 15 questions, add your own reflections, and instantly download a personalized PDF summary.")

# ---------- Data loading ----------
def load_questions(filename: str = "questions.json") -> Tuple[List[dict], List[str]]:
    base_dir = Path(__file__).parent
    path = base_dir / filename
    if not path.exists():
        st.error(f"Could not find {filename} at {path}. It must sit next to main/app.py.")
        try:
            st.caption("Directory listing for diagnostics:")
            for p in base_dir.iterdir():
                st.write("-", p.name)
        except Exception:
            pass
        st.stop()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"], data.get("themes", [])

# ---------- Logo loader (webp -> png without alpha for FPDF 1.x) ----------
def get_logo_png_path() -> Optional[str]:
    """Try several common locations. If WEBP found, convert to /tmp/logo.png without alpha."""
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
                    # FPDF 1.x can't handle PNG with alpha; flatten to RGB.
                    img = Image.open(p).convert("RGB")
                    out = Path("/tmp/logo.png")
                    img.save(out, format="PNG")
                    return str(out)
                except Exception:
                    return None
    return None

# ---------- PDF text safety (Latin-1 for classic fpdf) ----------
def safe_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # normalize curly punctuation
    s = (s.replace("’", "'")
         .replace("‘", "'")
         .replace("“", '"')
         .replace("”", '"')
         .replace("–", "-")
         .replace("—", "-")
         .replace("…", "...")
         .replace("•", "*"))
    # strip non-latin-1
    s = unicodedata.normalize("NFKD", s).encode("latin-1", "ignore").decode("latin-1")
    return s

# ---------- Optional AI ----------
USE_AI = bool(os.getenv("OPENAI_API_KEY"))

def pick_model(_: int, __: str) -> Tuple[str, int]:
    """Always use High model & cap (Deluxe)."""
    return HIGH_MODEL, MAX_TOK_HIGH

def ai_sections_and_weights(
    scores: Dict[str, int],
    top3: List[str],
    free_responses: List[dict],
    first_name: str,
    horizon_weeks: int = 4,
    depth_mode: str = "High",
) -> Optional[dict]:
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
        total_free_chars = sum(len(p.get("a", "")) for p in packed)
        model, max_tokens = pick_model(total_free_chars, depth_mode)

        prompt = (
            "You are a warm, practical life coach. Return ONLY valid JSON with keys:\n"
            "  archetype (string), core_need (string),\n"
            "  deep_insight (string, 220-380 words, address the user by first name),\n"
            "  why_now (string, 100-150 words),\n"
            "  strengths (array of 4-6 short strings),\n"
            "  energizers (array of 4), drainers (array of 4),\n"
            "  tensions (array of 2-3 short strings), blindspot (string <= 60 words),\n"
            "  actions (array of EXACTLY 3 short bullet strings),\n"
            "  if_then (array of EXACTLY 3 implementation-intention strings like: 'If it’s 7pm, then I…'),\n"
            "  weekly_plan (array of 7 brief day-plan strings),\n"
            "  affirmation (string <= 15 words), quote (string <= 20 words),\n"
            "  signature_metaphor (string <= 12 words), signature_sentence (string <= 20 words),\n"
            "  top_theme_boosters (array of up to 4 short suggestions), pitfalls (array of up to 4),\n"
            "  future_snapshot (string, 120-160 words, second-person, present tense, written AS IF it is {h} "
            "weeks later and the person followed through),\n"
            "  weights (object mapping question_id -> object of theme:int in [-2,2]).\n"
            f"User first name: {first_name or 'Friend'}.\n"
            f"Theme scores so far: {score_lines}.\n"
            f"Top 3 themes: {', '.join(top3)}.\n"
            f"Horizon weeks: {horizon_weeks}.\n"
            "Also consider these free-text answers (omit weights for questions you don't see):\n"
            f"{json.dumps(packed, ensure_ascii=False)}\n"
            "Tone: empathetic, encouraging, plain language. No medical/clinical claims. JSON only."
        ).format(h=horizon_weeks)

        resp = client.responses.create(
            model=model,
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": "Reply with helpful coaching guidance as STRICT JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_output_tokens=max_tokens,
        )
        try:
            u = getattr(resp, "usage", None)
            if u:
                st.caption(
                    f"AI model: {model} • usage input={getattr(u,'input_tokens','?')} "
                    f"output={getattr(u,'output_tokens','?')} total={getattr(u,'total_tokens','?')}"
                )
            else:
                st.caption(f"AI model: {model} • max_output_tokens={max_tokens}")
        except Exception:
            pass

        raw = resp.output_text or "{}"
        data = json.loads(raw)

        out = {
            "archetype": str(data.get("archetype", "")),
            "core_need": str(data.get("core_need", "")),
            "deep_insight": str(data.get("deep_insight", "")),
            "why_now": str(data.get("why_now", "")),
            "strengths": [str(x) for x in (data.get("strengths") or [])][:6],
            "energizers": [str(x) for x in (data.get("energizers") or [])][:4],
            "drainers": [str(x) for x in (data.get("drainers") or [])][:4],
            "tensions": [str(x) for x in (data.get("tensions") or [])][:3],
            "blindspot": str(data.get("blindspot", "")),
            "actions": [str(x) for x in (data.get("actions") or [])][:3],
            "if_then": [str(x) for x in (data.get("if_then") or [])][:3],
            "weekly_plan": [str(x) for x in (data.get("weekly_plan") or [])][:7],
            "affirmation": str(data.get("affirmation", "")),
            "quote": str(data.get("quote", "")),
            "signature_metaphor": str(data.get("signature_metaphor", "")),
            "signature_sentence": str(data.get("signature_sentence", "")),
            "top_theme_boosters": [str(x) for x in (data.get("top_theme_boosters") or [])][:4],
            "pitfalls": [str(x) for x in (data.get("pitfalls") or [])][:4],
            "future_snapshot": str(data.get("future_snapshot", "")),
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

# ---------- Pretty PDF (always safe_text) ----------
def draw_scores_barchart(pdf: FPDF, scores: Dict[str, int]):
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text("Your Theme Snapshot"), ln=True)
    pdf.set_font("Arial", "", 12)
    max_score = max(max(scores.values()), 1)
    bar_w_max = 120
    x_left = pdf.get_x() + 10
    y = pdf.get_y()
    for theme in THEMES:
        val = scores.get(theme, 0)
        bar_w = (val / max_score) * bar_w_max
        pdf.set_xy(x_left, y)
        pdf.cell(35, 6, safe_text(theme))
        pdf.set_fill_color(30, 144, 255)
        pdf.rect(x_left + 38, y + 1.3, bar_w, 4.5, "F")
        pdf.set_xy(x_left + 38 + bar_w + 2, y)
        pdf.cell(0, 6, safe_text(str(val)))
        y += 7
    pdf.set_y(y + 4)

def paragraph(pdf: FPDF, title: str, body: str):
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text(title), ln=True)
    pdf.set_font("Arial", "", 12)
    for line in safe_text(body).split("\n"):
        pdf.multi_cell(0, 6, line)
    pdf.ln(2)

def checkbox_line(pdf: FPDF, text: str):
    x = pdf.get_x()
    y = pdf.get_y()
    pdf.rect(x, y + 1.5, 4, 4)
    pdf.set_x(x + 6)
    pdf.multi_cell(0, 6, safe_text(text))

def label_value(pdf: FPDF, label: str, value: str):
    pdf.set_font("Arial", "B", 12); pdf.cell(0, 6, safe_text(label), ln=True)
    pdf.set_font("Arial", "", 12);  pdf.multi_cell(0, 6, safe_text(value))

def future_callout(pdf: FPDF, weeks: int, text: str):
    pdf.set_text_color(30, 60, 120)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text(f"Future Snapshot — {weeks} weeks"), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "I", 12)
    pdf.multi_cell(0, 6, safe_text(text))
    pdf.ln(2)

def make_pdf_bytes(first_name: str, email: str, scores: Dict[str,int], top3: List[str],
                   sections: dict, free_responses: List[dict], logo_path: Optional[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Logo (optional)
    if logo_path:
        try:
            pdf.image(logo_path, w=40); pdf.ln(2)
        except Exception:
            pass

    # Title block
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, safe_text(REPORT_TITLE), ln=True)
    pdf.set_font("Arial", "", 12)
    today = datetime.date.today().strftime("%d %b %Y")
    greet = f"Hi {first_name}," if first_name else "Hello,"
    pdf.cell(0, 8, safe_text(greet), ln=True)
    pdf.cell(0, 8, safe_text(f"Date: {today}"), ln=True)
    if email:
        pdf.cell(0, 8, safe_text(f"Email: {email}"), ln=True)
    pdf.ln(1)

    # Archetype/core & signature
    if sections.get("archetype") or sections.get("core_need"):
        label_value(pdf, "Archetype", sections.get("archetype","") or "—")
        label_value(pdf, "Core Need", sections.get("core_need","") or "—")
        if sections.get("signature_metaphor"):
            label_value(pdf, "Signature Metaphor", sections.get("signature_metaphor",""))
        if sections.get("signature_sentence"):
            label_value(pdf, "Signature Sentence", sections.get("signature_sentence",""))
        pdf.ln(1)

    # Top themes
    pdf.set_font("Arial", "B", 14); pdf.cell(0, 8, safe_text("Top Themes"), ln=True)
    pdf.set_font("Arial", "", 12);  pdf.multi_cell(0, 6, safe_text(", ".join(top3))); pdf.ln(1)

    # Score bars
    draw_scores_barchart(pdf, scores)

    # Insight blocks
    if sections.get("deep_insight"):
        paragraph(pdf, "What this really says about you", sections["deep_insight"])

    if sections.get("why_now"):
        label_value(pdf, "Why this matters now", sections["why_now"]); pdf.ln(1)

    # Future Snapshot callout
    if sections.get("future_snapshot"):
        future_callout(pdf, sections.get("horizon_weeks", 4), sections["future_snapshot"])

    # Strengths / Energizers / Drainers
    if sections.get("strengths"):
        pdf.set_font("Arial", "B", 14); pdf.cell(0, 8, "Signature strengths", ln=True)
        pdf.set_font("Arial", "", 12)
        for s in sections["strengths"]:
            pdf.cell(4, 6, "*"); pdf.multi_cell(0, 6, safe_text(s))
        pdf.ln(1)

    if sections.get("energizers") or sections.get("drainers"):
        pdf.set_font("Arial", "B", 14); pdf.cell(0, 8, "Energy map", ln=True)
        pdf.set_font("Arial", "B", 12); pdf.cell(0, 6, "Energizers", ln=True)
        pdf.set_font("Arial", "", 12)
        for e in sections.get("energizers", []):
            pdf.cell(4, 6, "+"); pdf.multi_cell(0, 6, safe_text(e))
        pdf.ln(1)
        pdf.set_font("Arial", "B", 12); pdf.cell(0, 6, "Drainers", ln=True)
        pdf.set_font("Arial", "", 12)
        for d in sections.get("drainers", []):
            pdf.cell(4, 6, "-"); pdf.multi_cell(0, 6, safe_text(d))
        pdf.ln(1)

    # Tensions & blindspot
    if sections.get("tensions"):
        pdf.set_font("Arial", "B", 14); pdf.cell(0, 8, "Hidden tensions", ln=True)
        pdf.set_font("Arial", "", 12)
        for t in sections["tensions"]:
            pdf.cell(4, 6, "*"); pdf.multi_cell(0, 6, safe_text(t))
        pdf.ln(1)
    if sections.get("blindspot"):
        label_value(pdf, "Watch-out (gentle blind spot)", sections["blindspot"]); pdf.ln(1)

    # Actions & If–Then plans
    if sections.get("actions"):
        pdf.set_font("Arial", "B", 14); pdf.cell(0, 8, "3 next-step actions (7 days)", ln=True)
        pdf.set_font("Arial", "", 12)
        for a in sections["actions"]:
            checkbox_line(pdf, a)
        pdf.ln(1)

    if sections.get("if_then"):
        pdf.set_font("Arial", "B", 14); pdf.cell(0, 8, "Implementation intentions (If–Then)", ln=True)
        pdf.set_font("Arial", "", 12)
        for it in sections.get("if_then", []):
            pdf.cell(4, 6, "*"); pdf.multi_cell(0, 6, safe_text(it))
        pdf.ln(1)

    # Weekly plan
    if sections.get("weekly_plan"):
        pdf.set_font("Arial", "B", 14); pdf.cell(0, 8, "1-week gentle plan", ln=True)
        pdf.set_font("Arial", "", 12)
        for i, item in enumerate(sections["weekly_plan"][:7]):
            pdf.cell(0, 6, safe_text(f"Day {i+1}: {item}"), ln=True)
        pdf.ln(1)

    # Balancing Opportunity (lowest 1–2 themes)
    lows = [name for name, _ in sorted(scores.items(), key=lambda x: x[1])[:2]]
    if lows:
        pdf.set_font("Arial", "B", 14); pdf.cell(0, 8, "Balancing Opportunity", ln=True)
        pdf.set_font("Arial", "", 12)
        for theme in lows:
            tip = balancing_suggestion(theme)
            pdf.multi_cell(0, 6, safe_text(f"{theme}: {tip}"))
        pdf.ln(1)

    # Boosters & Pitfalls (for top themes)
    if sections.get("top_theme_boosters") or sections.get("pitfalls"):
        pdf.set_font("Arial", "B", 14); pdf.cell(0, 8, "Amplify what works / Avoid what trips you", ln=True)
        if sections.get("top_theme_boosters"):
            pdf.set_font("Arial", "B", 12); pdf.cell(0, 6, "Boosters", ln=True)
            pdf.set_font("Arial", "", 12)
            for b in sections.get("top_theme_boosters", []):
                pdf.cell(4, 6, "*"); pdf.multi_cell(0, 6, safe_text(b))
        if sections.get("pitfalls"):
            pdf.set_font("Arial", "B", 12); pdf.cell(0, 6, "Pitfalls", ln=True)
            pdf.set_font("Arial", "", 12)
            for p in sections.get("pitfalls", []):
                pdf.cell(4, 6, "-"); pdf.multi_cell(0, 6, safe_text(p))
        pdf.ln(1)

    # Quote & affirmation
    if sections.get("affirmation") or sections.get("quote"):
        pdf.set_font("Arial", "B", 12); pdf.cell(0, 6, "Keep this in view", ln=True)
        pdf.set_font("Arial", "I", 11)
        if sections.get("affirmation"):
            pdf.multi_cell(0, 6, safe_text(f"Affirmation: {sections['affirmation']}"))
        if sections.get("quote"):
            qtext = f'"{sections["quote"]}"'
            pdf.multi_cell(0, 6, safe_text(qtext))
        pdf.ln(2)

    # Your reflections (free text)
    if free_responses:
        pdf.set_font("Arial", "B", 14); pdf.cell(0, 8, "Your words we heard", ln=True)
        pdf.set_font("Arial", "", 12)
        for fr in free_responses:
            if not fr.get("answer"):
                continue
            pdf.multi_cell(0, 6, safe_text(f"• {fr['question']}"))
            pdf.multi_cell(0, 6, safe_text(f"  {fr['answer']}"))
            pdf.ln(1)

    # Clear cue before checklist page
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 6, safe_text("On the next page: a printable ‘Signature Week — At a glance’ checklist you can use right away."))

    # New page: Signature Week (at a glance)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, safe_text("Signature Week — At a glance"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, safe_text("A simple plan you can print or screenshot. Check items off as you go."))
    pdf.ln(2)

    # 7-row checklist for the weekly_plan (or fall back)
    week_items = sections.get("weekly_plan") or []
    if not week_items:
        week_items = [f"Do one small action for {t}" for t in top3] + ["Reflect and set next step"]

    for i, item in enumerate(week_items[:7]):
        x = pdf.get_x()
        y = pdf.get_y()
        pdf.rect(x, y + 1.5, 4, 4)
        pdf.set_x(x + 6)
        pdf.multi_cell(0, 6, safe_text(f"Day {i+1}: {item}"))

    # Tiny Progress Tracker
    pdf.ln(2)
    pdf.set_font("Arial", "B", 14); pdf.cell(0, 8, safe_text("Tiny Progress Tracker"), ln=True)
    pdf.set_font("Arial", "", 12)
    milestones = sections.get("actions") or [
        "Choose one tiny step and schedule it.",
        "Tell a friend your plan for gentle accountability.",
        "Spend 20 minutes on your step and celebrate completion."
    ]
    for m in milestones[:3]:
        x = pdf.get_x()
        y = pdf.get_y()
        pdf.rect(x, y + 1.5, 4, 4)
        pdf.set_x(x + 6)
        pdf.multi_cell(0, 6, safe_text(m))
    pdf.ln(2)

    # Footer
    pdf.set_font("Arial", "I", 10); pdf.ln(2)
    pdf.multi_cell(0, 5, safe_text("Life Minus Work • This report is a starting point for reflection. Nothing here is medical or financial advice."))
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
    answers: Dict[str, dict] = {}
    free_responses: List[dict] = []

    # Personalization (just the horizon; depth forced to High)
    with st.expander("Personalization options"):
        horizon_weeks = st.slider("Future snapshot horizon (weeks)", 2, 8, 4)
    depth_mode = "High"  # forced
    st.session_state["depth"] = "high"

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
            if choice_idx == len(options) - 1:
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

        # AI sections + optional weights from free text
        sections = {"summary": "", "actions": [], "weekly_plan": [], "weights": {}}
        if USE_AI:
            maybe = ai_sections_and_weights(
                scores,
                top_themes(scores),
                free_responses,
                st.session_state.get("first_name", ""),
                horizon_weeks=horizon_weeks,
                depth_mode=depth_mode,
            )
            if maybe:
                sections.update(maybe)
                if sections.get("weights"):
                    scores = apply_free_text_weights(scores, sections["weights"])
                sections["horizon_weeks"] = horizon_weeks

        # Fallback if AI off/failed
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
            fsnap = (
                f"It’s {horizon_weeks} weeks later. You’ve stayed close to what matters, "
                f"protecting time for {top_themes(scores)[0] if top_themes(scores) else 'what energizes you'}. "
                f"A few tiny actions, repeated, build confidence. You pause, adjust, and keep going."
            )
            sections = {
                "deep_insight": base,
                "actions": actions,
                "weekly_plan": plan,
                "future_snapshot": fsnap,
                "horizon_weeks": horizon_weeks,
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
            st.session_state.get("first_name", ""),
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
                    st.session_state.get("first_name", ""),
                    email,
                    json.dumps(scores),
                    json.dumps(top3),
                ])
            st.caption("Saved to /tmp/responses.csv (Cloud-safe, ephemeral).")
        except Exception as e:
            st.caption(f"Could not save responses (demo only). {e}")
else:
    st.info("Start by entering your first name above.")
