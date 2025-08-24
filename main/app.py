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
OPENAI_MODEL = "gpt-5-nano"  # or "gpt-4o-mini"

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

# ---------- PDF text safety (FPDF core fonts are Latin-1) ----------
def safe_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = (s.replace("\u2019", "'")   # ’
           .replace("\u2018", "'")  # ‘
           .replace("\u201c", '"')  # “
           .replace("\u201d", '"')  # ”
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
    """
    Ask the model for:
      - summary (140–220 words), actions[3], weekly_plan[7]
      - weights: mapping of question_id -> {theme: -2..+2} for free-text answers
    Returns dict with keys: summary, actions, weekly_plan, weights
    """
    if not USE_AI:
        return None
    try:
        from openai import OpenAI
        client = OpenAI()

        score_lines = ", ".join([f"{k}: {v}" for k, v in scores.items()])
        # Pack free responses compactly for lower token use
        packed = [
            {"id": fr["id"], "q": fr["question"], "a": fr["answer"][:280]}
            for fr in free_responses if fr.get("answer")
        ]

        prompt = (
            "You are a warm, practical life coach. Return ONLY valid JSON with keys:\n"
            "  summary (string, 140-220 words, address the user by first name),\n"
            "  actions (array of EXACTLY 3 short bullet strings),\n"
            "  weekly_plan (array of 7 brief day-plan strings),\n"
            "  weights (object mapping question_id -> object of theme:int scores in [-2,2]).\n"
            f"User first name: {first_name or 'Friend'}.\n"
            f"Theme scores so far: {score_lines}.\n"
            f"Top 3 themes: {', '.join(top3)}.\n"
            "Also consider these free-text answers. If a question isn't in the list, omit it from weights.\n"
            f"Free responses: {json.dumps(packed, ensure_ascii=False)}\n"
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
            max_output_tokens=900,
        )
        raw = resp.output_text or "{}"
        data = json.loads(raw)

        # normalize shapes
        out = {
            "summary": str(data.get("summary", "")),
            "actions": [str(a) for a in (data.get("actions") or [])][:3],
            "weekly_plan": [str(a) for a in (data.get("weekly_plan") or [])][:7],
            "weights": {},
        }
        weights = data.get("weights") or {}
        # ensure only known themes and int in [-2,2]
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

# ---------- Pretty PDF ----------
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

def bullets(pdf: FPDF, title: str, items: List[str]):
    if not items: return
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text(title), ln=True)
    pdf.set_font("Arial", "", 12)
    for it in items:
        pdf.cell(4, 6, "*")
        pdf.multi_cell(0, 6, safe_text(it))
    pdf.ln(1)

def make_pdf_bytes(first_name: str, email: str, scores: Dict[str,int], top3: List[str],
                   sections: dict, free_responses: List[dict], logo_path: Optional[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Logo (optional)
    if logo_path:
        try:
            pdf.image(logo_path, w=40)
            pdf.ln(2)
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
    pdf.ln(2)

    # Top themes
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text("Top Themes"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, safe_text(", ".join(top3)))
    pdf.ln(2)

    # Bars
    draw_scores_barchart(pdf, scores)

    # Narrative
    paragraph(pdf, "Personalized Summary", sections.get("summary", ""))
    bullets(pdf, "Three Next-Step Actions (7 days)", sections.get("actions", []))
    weekly = [f"Day {i+1}: {t}" for i, t in enumerate(sections.get("weekly_plan", []))]
    bullets(pdf, "1-Week Gentle Plan", weekly)

    # Your reflections (free text)
    if free_responses:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, safe_text("Your Reflections"), ln=True)
        pdf.set_font("Arial", "", 12)
        for fr in free_responses:
            if not fr.get("answer"): continue
            pdf.multi_cell(0, 6, safe_text(f"• {fr['question']}"))
            pdf.multi_cell(0, 6, safe_text(f"  {fr['answer']}"))
            pdf.ln(1)

    # Footer
    pdf.set_font("Arial", "I", 10)
    pdf.ln(2)
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
    # answers dict: qid -> {"choice_idx": int|None, "free_text": str|None}
    answers: Dict[str, dict] = {}
    free_responses: List[dict] = []

    for q in questions:
        st.subheader(q["text"])
        options = [c["label"] for c in q["choices"]] + ["✍️ I’ll write my own answer"]
        # radio returns the label; use a unique key per question
        selected = st.radio("Choose one:", options, index=None, key=f"{q['id']}_choice")
        choice_idx = None
        free_text_val = None

        if selected == "✍️ I’ll write my own answer":
            free_text_val = st.text_area("Your answer", key=f"{q['id']}_free", height=80, placeholder="Type your own response...")
        elif selected is not None:
            choice_idx = options.index(selected)
            # adjust because we appended the custom option at the end
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

        # If AI available, ask for sections + weights for free answers
        sections = {"summary": "", "actions": [], "weekly_plan": [], "weights": {}}
        if USE_AI:
            maybe = ai_sections_and_weights(scores, top_themes(scores), free_responses, st.session_state.get("first_name",""))
            if maybe:
                sections.update(maybe)
                if sections.get("weights"):
                    scores = apply_free_text_weights(scores, sections["weights"])

        # Fallback sections if AI off or failed
        if not sections.get("summary"):
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
            sections = {"summary": base, "actions": actions, "weekly_plan": plan, "weights": {}}

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
