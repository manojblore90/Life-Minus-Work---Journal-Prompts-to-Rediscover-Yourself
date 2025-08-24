import os
import sys
import json
import datetime
import unicodedata
from pathlib import Path
import streamlit as st
from fpdf import FPDF

APP_TITLE = "Life Minus Work — Reflection Quiz (15 questions)"
REPORT_TITLE = "Your Reflection Report"
THEMES = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="centered")
st.title(APP_TITLE)
st.caption("🩺 Diagnostic: App is running. Use the sidebar to verify files.")

# Diagnostics
st.sidebar.header("Diagnostics")
st.sidebar.write("Python:", sys.version)
st.sidebar.write("Working dir:", os.getcwd())
st.sidebar.write("File location:", Path(__file__).parent)
st.sidebar.write("Env has OPENAI_API_KEY:", bool(os.getenv("OPENAI_API_KEY")))
try:
    st.sidebar.write("Files here:", [p.name for p in Path(__file__).parent.iterdir()])
except Exception as e:
    st.sidebar.error(f"Dir list error: {e}")

def load_questions(filename="questions.json"):
    base_dir = Path(__file__).parent
    path = base_dir / filename
    if not path.exists():
        st.error(f"Could not find {filename} at {path}. It must be next to app.py in the 'main' folder.")
        st.stop()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"], data.get("themes", [])

# ---- PDF safety: sanitize text to Latin-1 (FPDF core fonts are Latin-1 only) ----
def safe_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # Keep common punctuation by mapping to ASCII
    s = (s.replace("\u2019", "'")  # right single quote
           .replace("\u2018", "'")  # left single quote
           .replace("\u201c", '"')  # left double quote
           .replace("\u201d", '"')  # right double quote
           .replace("\u2013", "-")  # en dash
           .replace("\u2014", "-")) # em dash
    # Normalize and strip any remaining unsupported chars
    s = unicodedata.normalize("NFKD", s).encode("latin-1", "ignore").decode("latin-1")
    return s

# Optional AI
USE_AI = bool(os.getenv("OPENAI_API_KEY"))
if USE_AI:
    try:
        from openai import OpenAI
        openai_client = OpenAI()
    except Exception:
        USE_AI = False

def compute_scores(answers, questions):
    scores = {t: 0 for t in THEMES}
    for q in questions:
        qid = q["id"]
        choice_idx = answers.get(qid)
        if choice_idx is None:
            continue
        try:
            choice = q["choices"][choice_idx]
        except IndexError:
            continue
        for theme, val in choice.get("weights", {}).items():
            scores[theme] = scores.get(theme, 0) + val
    return scores

def top_themes(scores, k=3):
    return [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

def ai_paragraph(prompt):
    if not USE_AI:
        return None
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a warm, practical life coach. Be concise and supportive."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None

def generate_report_text(email, scores, top3):
    base_copy = [
        "Thank you for completing the Reflection Quiz. Below are your top themes and next-step ideas tailored for you.",
        *[f"- {theme}: Consider one simple action this week to build momentum in this area." for theme in top3],
        "Tip: Small consistent actions beat big one-off efforts. Be kind to yourself as you experiment.",
    ]
    fallback = "\n".join(base_copy)

    if USE_AI:
        score_lines = ", ".join([f"{k}: {v}" for k, v in scores.items()])
        prompt = (
            "Create a friendly, empowering summary (140-200 words) for a user with these theme scores: "
            f"{score_lines}.\n"
            f"Top 3 themes: {', '.join(top3)}.\n"
            "Voice: empathetic, practical, and encouraging; avoid medical claims.\n"
            "Give 3 short bullet-point actions for the next 7 days, tailored to the themes.\n"
            "Do not mention scores. Address the reader as 'you'."
        )
        ai_text = ai_paragraph(prompt)
        if ai_text:
            return ai_text
    return fallback

def make_pdf_bytes(name_email, scores, top3, narrative):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, safe_text(REPORT_TITLE), ln=True)
    pdf.set_font("Arial", "", 12)
    today = datetime.date.today().strftime("%d %b %Y")
    pdf.cell(0, 8, safe_text(f"Date: {today}"), ln=True)
    if name_email:
        pdf.cell(0, 8, safe_text(f"Email: {name_email}"), ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text("Your Theme Snapshot"), ln=True)
    pdf.set_font("Arial", "", 12)
    for t in THEMES:
        pdf.cell(0, 7, safe_text(f"- {t}: {scores.get(t, 0)}"), ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text("Top Themes"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, safe_text(", ".join(top3)))
    pdf.ln(2)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text("Your Personalized Guidance"), ln=True)
    pdf.set_font("Arial", "", 12)
    for line in safe_text(narrative).split("\n"):
        pdf.multi_cell(0, 6, line)

    pdf.ln(6)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 6, safe_text("Life Minus Work • This report is a starting point for reflection. Nothing here is medical or financial advice."))

    # Return bytes for Streamlit download
    return pdf.output(dest="S").encode("latin-1")

st.divider()
st.subheader("Step 1: Start here")
with st.form("email_form"):
    email = st.text_input("Your email (to save your results and for your download link)", placeholder="you@example.com")
    consent = st.checkbox("I agree to receive my results and occasional updates from Life Minus Work.")
    submitted = st.form_submit_button("Start Quiz")
    if submitted and (not email or not consent):
        st.error("Please enter your email and give consent to continue.")

if 'submitted_once' not in st.session_state:
    st.session_state['submitted_once'] = False
if submitted and email and consent:
    st.session_state["email"] = email
    st.session_state['submitted_once'] = True
    st.success("Great! Scroll down to begin.")

st.divider()
st.subheader("Step 2: Questions")
if st.session_state.get('submitted_once'):
    questions, _ = load_questions("questions.json")
    st.caption("You can scroll and answer at your own pace.")

    answers = {}
    for q in questions:
        st.markdown(f"**{q['text']}**")
        options = [c["label"] for c in q["choices"]]
        choice = st.radio("Choose one:", options, index=None, key=q["id"])
        if choice is not None:
            answers[q["id"]] = options.index(choice)
        st.divider()

    st.subheader("Step 3: Get your report")
    if len(answers) < len(questions]):
        st.info(f"Answered {len(answers)} of {len(questions)} questions. Keep going!")
    else:
        if st.button("Finish and Generate My Report"):
            scores = compute_scores(answers, questions)
            top3 = top_themes(scores, 3)
            narrative = generate_report_text(st.session_state.get('email',''), scores, top3)
            pdf_bytes = make_pdf_bytes(st.session_state.get('email',''), scores, top3, narrative)

            st.success("Your personalized report is ready!")
            st.download_button("Download Your PDF Report", data=pdf_bytes,
                               file_name="LifeMinusWork_Reflection_Report.pdf", mime="application/pdf")

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
                    writer.writerow([ts, st.session_state.get('email',''), json.dumps(scores), json.dumps(top3)])
                st.caption("Saved to /tmp/responses.csv (Cloud-safe, ephemeral).")
            except Exception as e:
                st.caption(f"Could not save responses (demo only). {e}")
else:
    st.info("Enter your email above and tick the consent box to begin.")
