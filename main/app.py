import os
import json
import datetime
import unicodedata
from pathlib import Path
import streamlit as st
from fpdf import FPDF

APP_TITLE = "Life Minus Work — Reflection Quiz (15 questions)"
REPORT_TITLE = "Your Reflection Report"
THEMES = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

# ---- Choose your model here ----
OPENAI_MODEL = "gpt-5-nano"   # or "gpt-4o-mini"

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="centered")
st.title(APP_TITLE)
st.write("Answer 15 questions and instantly download a personalized PDF summary.")

# ------------ Data loading ------------
def load_questions(filename: str = "questions.json"):
    base_dir = Path(__file__).parent
    path = base_dir / filename
    if not path.exists():
        st.error(f"Could not find {filename} at {path}. It must sit next to main/app.py.")
        st.stop()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"], data.get("themes", [])

# ------------ PDF text safety (FPDF core fonts are Latin-1) ------------
def safe_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # Map common curly punctuation to ASCII so FPDF doesn't crash
    s = (s.replace("\u2019", "'")   # ’
           .replace("\u2018", "'")  # ‘
           .replace("\u201c", '"')  # “
           .replace("\u201d", '"')  # ”
           .replace("\u2013", "-")  # –
           .replace("\u2014", "-")) # —
    # Drop anything Latin-1 can't handle
    s = unicodedata.normalize("NFKD", s).encode("latin-1", "ignore").decode("latin-1")
    return s

# ------------ Optional AI (set OPENAI_API_KEY in Secrets) ------------
USE_AI = bool(os.getenv("OPENAI_API_KEY"))
def ai_report_sections(scores: dict, top3: list) -> dict | None:
    if not USE_AI:
        return None
    try:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from environment/Secrets

        score_lines = ", ".join([f"{k}: {v}" for k, v in scores.items()])
        prompt = (
            "You are a warm, practical life coach. Return ONLY valid JSON with keys: "
            'summary (string, 140-220 words), actions (array of exactly 3 short bullet strings), '
            "weekly_plan (array of 7 brief day-plan strings). "
            f"User theme scores: {score_lines}. Top 3 themes: {', '.join(top3)}. "
            "Tone: empathetic, encouraging, plain language. "
            "No medical/clinical claims. Do not include extra commentary outside the JSON."
        )

        resp = client.responses.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": "Reply with concise, helpful coaching guidance as valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_output_tokens=500,
        )
        raw = resp.output_text or "{}"
        data = json.loads(raw)
        # normalize shape
        return {
            "summary": str(data.get("summary", "")),
            "actions": [str(a) for a in (data.get("actions") or [])][:3],
            "weekly_plan": [str(a) for a in (data.get("weekly_plan") or [])][:7],
        }
    except Exception:
        return None

# ------------ Scoring helpers ------------
def compute_scores(answers: dict, questions: list) -> dict:
    scores = {t: 0 for t in THEMES}
    for q in questions:
        qid = q["id"]
        choice_idx = answers.get(qid)
        if choice_idx is None:
            continue
        try:
            choice = q["choices"][choice_idx]
        except (IndexError, KeyError, TypeError):
            continue
        for theme, val in choice.get("weights", {}).items():
            scores[theme] = scores.get(theme, 0) + val
    return scores

def top_themes(scores: dict, k: int = 3) -> list:
    return [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

# ------------ Pretty PDF helpers (bars + sections) ------------
def draw_scores_barchart(pdf: FPDF, scores: dict):
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text("Your Theme Snapshot"), ln=True)
    pdf.set_font("Arial", "", 12)
    max_score = max(max(scores.values()), 1)
    bar_w_max = 120  # width of the longest bar
    x_left = pdf.get_x() + 10
    y = pdf.get_y()
    for theme in THEMES:
        val = scores.get(theme, 0)
        bar_w = (val / max_score) * bar_w_max
        # label
        pdf.set_xy(x_left, y)
        pdf.cell(35, 6, safe_text(theme))
        # bar
        pdf.set_fill_color(30, 144, 255)  # blue
        pdf.rect(x_left + 38, y + 1.3, bar_w, 4.5, "F")
        # value
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

def bullets(pdf: FPDF, title: str, items: list):
    if not items:
        return
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text(title), ln=True)
    pdf.set_font("Arial", "", 12)
    for it in items:
        pdf.cell(4, 6, "*")
        pdf.multi_cell(0, 6, safe_text(it))
    pdf.ln(1)

def make_pdf_bytes(email: str, scores: dict, top3: list, sections: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title block
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, safe_text(REPORT_TITLE), ln=True)
    pdf.set_font("Arial", "", 12)
    today = datetime.date.today().strftime("%d %b %Y")
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

    # Narrative sections
    paragraph(pdf, "Personalized Summary", sections.get("summary", ""))
    bullets(pdf, "Three Next-Step Actions (7 days)", sections.get("actions", []))
    weekly = [f"Day {i+1}: {t}" for i, t in enumerate(sections.get("weekly_plan", []))]
    bullets(pdf, "1-Week Gentle Plan", weekly)

    # Footer
    pdf.set_font("Arial", "I", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 5, safe_text("Life Minus Work • This report is a starting point for reflection. Nothing here is medical or financial advice."))

    return pdf.output(dest="S").encode("latin-1")

# ------------ Non-AI fallback ------------
def fallback_sections(scores: dict, top3: list) -> dict:
    base = "Thank you for completing the Reflection Quiz. Below are your top themes and next-step ideas tailored for you."
    actions = [
        f"Choose one tiny step to honor {top3[0].lower()} this week.",
        "Tell a friend your plan—gentle accountability.",
        "Schedule 20 minutes for reflection or journaling."
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
    return {"summary": base, "actions": actions, "weekly_plan": plan}

# ------------ UI ------------
with st.form("email_form"):
    email = st.text_input("Your email (for your download link)", placeholder="you@example.com")
    consent = st.checkbox("I agree to receive my results and occasional updates from Life Minus Work.")
    submitted = st.form_submit_button("Start Quiz")
    if submitted and (not email or not consent):
        st.error("Please enter your email and give consent to continue.")

if "submitted_once" not in st.session_state:
    st.session_state["submitted_once"] = False
if submitted and email and consent:
    st.session_state["email"] = email
    st.session_state["submitted_once"] = True
    st.success("Great! Scroll down to begin.")

if st.session_state.get("submitted_once"):
    st.header("Your Questions")
    st.caption("You can scroll and answer at your own pace.")

    questions, _ = load_questions("questions.json")
    answers = {}

    for q in questions:
        st.subheader(q["text"])
        options = [c["label"] for c in q["choices"]]
        choice = st.radio("Choose one:", options, index=None, key=q["id"])
        if choice is not None:
            answers[q["id"]] = options.index(choice)
        st.divider()

    st.subheader("Step 3: Get your report")
    if len(answers) < len(questions):
        st.info(f"Answered {len(answers)} of {len(questions)} questions. Keep going!")
    else:
        if st.button("Finish and Generate My Report"):
            scores = compute_scores(answers, questions)
            top3 = top_themes(scores, 3)

            # AI sections or fallback
            sections = ai_report_sections(scores, top3) if USE_AI else None
            if sections is None:
                sections = fallback_sections(scores, top3)

            pdf_bytes = make_pdf_bytes(st.session_state.get("email", ""), scores, top3, sections)

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
                        writer.writerow(["timestamp", "email", "scores", "top3"])
                    writer.writerow([ts, st.session_state.get("email", ""), json.dumps(scores), json.dumps(top3)])
                st.caption("Saved to /tmp/responses.csv (Cloud-safe, ephemeral).")
            except Exception as e:
                st.caption(f"Could not save responses (demo only). {e}")
else:
    st.info("Enter your email above and tick the consent box to begin.")
