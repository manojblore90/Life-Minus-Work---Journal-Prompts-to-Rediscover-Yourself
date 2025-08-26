# app.py  — Life Minus Work (full cleaned build, Unicode PDF + robust AI)
# ----------------------------------------------------------------------------------
import os, sys, re, json, unicodedata, datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
from PIL import Image
from fpdf import FPDF

# OpenAI SDK (safe import if key missing)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------------
APP_TITLE   = "Life Minus Work — Reflection Quiz (15 questions)"
REPORT_TITLE = "Your Reflection Report"
THEMES       = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="centered")

# Secrets / env
def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

USE_AI = bool(OPENAI_API_KEY and OpenAI)
HIGH_MODEL = get_secret("OPENAI_HIGH_MODEL", "gpt-4o-mini")  # Stable model
MAX_TOK_HIGH = int(get_secret("MAX_OUTPUT_TOKENS_HIGH", "8000"))
FALLBACK_CAP = int(get_secret("MAX_OUTPUT_TOKENS_FALLBACK", "6000"))

# ----------------------------------------------------------------------------------
# DIAGNOSTICS (can be collapsed later)
# ----------------------------------------------------------------------------------
with st.expander("🔧 Diagnostics (temporary)", expanded=False):
    st.write("Python:", sys.version.split()[0])
    st.write("__file__:", __file__)
    st.write("cwd:", os.getcwd())
    here = Path(__file__).parent
    try:
        st.write("Directory listing next to app.py:", [p.name for p in here.iterdir()])
    except Exception as e:
        st.write("Dir list failed:", e)
    try:
        import fpdf as _fp
        st.write("fpdf2 version:", getattr(_fp, "__version__", "unknown"))
    except Exception:
        pass
    masked = (OPENAI_API_KEY[:4] + "…" + OPENAI_API_KEY[-4:]) if OPENAI_API_KEY else "None"
    st.write("OPENAI_API_KEY detected:", bool(OPENAI_API_KEY), "| key:", masked if OPENAI_API_KEY else "—")
    st.write("Model:", HIGH_MODEL, "| MAX_TOK_HIGH:", MAX_TOK_HIGH, "| FALLBACK_CAP:", FALLBACK_CAP)
    if st.button("Probe: load questions.json"):
        p = here / "questions.json"
        st.write("Path:", str(p), "| exists:", p.exists())
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                st.success(f"Loaded {len(data.get('questions', []))} questions.")
            except Exception as e:
                st.error(f"JSON load error: {e}")

# ----------------------------------------------------------------------------------
# TEXT CLEANING & PDF-SAFE MULTICELL
# ----------------------------------------------------------------------------------
def _ascii_only(s: str) -> str:
    # Map fancy punctuation to ASCII
    return (s.replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"')
             .replace("–", "-").replace("—", "-")
             .replace("…", "...").replace("•", "*"))

def clean_text(s: str, max_len: int = 50, ascii_fallback: bool = False) -> str:
    """Bulletproof text cleaner for PDF output."""
    if not s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    # Remove control chars except newline
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    # Split on any whitespace, preserving it
    pieces, out = re.split(r"(\s+)", s), []
    for t in pieces:
        if t.strip():  # Non-whitespace
            while len(t) > max_len:
                out.append(t[:max_len] + "...")
                t = t[max_len:]
            out.append(t)
        else:
            out.append(t)  # Keep original whitespace
    s = "".join(out)
    if ascii_fallback:
        s = _ascii_only(s)
    return s

def mc(pdf: "FPDF", text: str, h: float = 6, unicode_ok: bool = True):
    """
    Safe MultiCell:
    - Cleans text
    - If not unicode_ok (Helvetica), convert to ASCII-safe punctuation
    - Never crashes; absolute fallback prints an ASCII notice
    """
    s = clean_text(text or "", ascii_fallback=not unicode_ok)
    try:
        pdf.multi_cell(0, h, s)
    except Exception:
        try:
            for line in s.split("\n"):
                pdf.multi_cell(0, h, clean_text(line, ascii_fallback=not unicode_ok))
        except Exception:
            pdf.multi_cell(0, h, "[...content truncated...]")  # ASCII-only

# ----------------------------------------------------------------------------------
# UNICODE PDF (TTF) LOADER
# ----------------------------------------------------------------------------------
def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return str(p)
    return None

def create_pdf_with_unicode() -> Tuple[FPDF, bool]:
    """
    Create FPDF. If a TTF font is found, register it and return (pdf, True),
    else fall back to Helvetica and return (pdf, False).
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=15, top=15, right=15)  # Increased margins

    here = Path(__file__).parent
    candidates_regular = [
        here / "DejaVuSans.ttf",
        here / "assets" / "DejaVuSans.ttf",
        here / "assets" / "fonts" / "DejaVuSans.ttf",
    ]
    candidates_bold = [
        here / "DejaVuSans-Bold.ttf",
        here / "assets" / "DejaVuSans-Bold.ttf",
        here / "assets" / "fonts" / "DejaVuSans-Bold.ttf",
    ]
    candidates_italic = [
        here / "DejaVuSans-BoldOblique.ttf",
        here / "assets" / "DejaVuSans-BoldOblique.ttf",
        here / "assets" / "fonts" / "DejaVuSans-BoldOblique.ttf",
    ]
    reg = _first_existing(candidates_regular)
    if reg:
        try:
            pdf.add_font("LMW", "", reg, uni=True)
            b = _first_existing(candidates_bold)
            if b:
                try:
                    pdf.add_font("LMW", "B", b, uni=True)
                except Exception:
                    pass
            i = _first_existing(candidates_italic)
            if i:
                try:
                    pdf.add_font("LMW", "I", i, uni=True)
                except Exception:
                    pass
            pdf.set_font("LMW", "", 12)
            return pdf, True
        except Exception:
            pass

    # Fallback (Latin‑1 only)
    pdf.set_font("Helvetica", "", 12)
    return pdf, False

def setf(pdf: FPDF, unicode_ok: bool, style: str = "", size: int = 12):
    """Smart set_font across Unicode/Helvetica paths."""
    if unicode_ok:
        try:
            pdf.set_font("LMW", style or "", size)
        except Exception:
            pdf.set_font("LMW", "", size)
    else:
        pdf.set_font("Helvetica", style or "", size)

# ----------------------------------------------------------------------------------
# LOGO LOADER
# ----------------------------------------------------------------------------------
def get_logo_png_path() -> Optional[str]:
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
                    img = Image.open(p).convert("RGB")
                    out = Path("/tmp/logo.png")
                    img.save(out, format="PNG")
                    return str(out)
                except Exception:
                    return None
    return None

# ----------------------------------------------------------------------------------
# QUESTIONS LOADER
# ----------------------------------------------------------------------------------
def load_questions(filename="questions.json"):
    base_dir = Path(__file__).parent
    path = base_dir / filename
    if not path.exists():
        st.error(f"Could not find {filename} at {path}. It must sit next to app.py.")
        try:
            st.caption("Directory listing:")
            for p in base_dir.iterdir():
                st.write("-", p.name)
        except Exception:
            pass
        st.stop()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"], data.get("themes", [])

# ----------------------------------------------------------------------------------
# SCORING
# ----------------------------------------------------------------------------------
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
        "Identity": "Choose one tiny ritual that reflects who you are becoming.",
        "Growth": "Pick a single skill and block 15 minutes to practice today.",
        "Connection": "Send a 3-line check-in to someone who matters.",
        "Peace": "Name a 10-minute wind-down you will repeat daily.",
        "Adventure": "Plan a 30–60 minute micro-adventure within 7 days.",
        "Contribution": "Offer one concrete act of help this week.",
    }
    return suggestions.get(theme, "Take one small, visible step this week.")

# ----------------------------------------------------------------------------------
# OPENAI CALL (defensive; JSON-only; token tracker)
# ----------------------------------------------------------------------------------
def _call_openai_json(model: str, system: str, user: str, cap: int):
    """
    Try Chat Completions JSON mode with max_completion_tokens.
    Return (content, usage, path_label)
    """
    if not (USE_AI and OpenAI):
        raise RuntimeError("OpenAI not configured")
    client = OpenAI()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=cap,
            response_format={"type": "json_object"},
        )
        content = r.choices[0].message.content if r.choices else ""
        usage = getattr(r, "usage", None)
        usage_dict = None
        if usage is not None:
            usage_dict = {
                "input": getattr(usage, "prompt_tokens", None),
                "output": getattr(usage, "completion_tokens", None),
                "total": getattr(usage, "total_tokens", None),
            }
        return content, usage_dict, "chat+rf_mct"
    except Exception as e:
        st.warning(f"OpenAI API call failed: {str(e)}. Falling back to default template.")
        return None, None, None

def ai_sections_and_weights(
    scores: Dict[str, int],
    top3: List[str],
    free_responses: List[dict],
    first_name: str,
    horizon_weeks: int = 4,
) -> Optional[dict]:
    if not USE_AI:
        return None
    try:
        system_prompt = """
        You are an expert assistant creating a personalized reflection report. Based on the user's quiz scores, top themes, and free-text responses, generate a JSON object with the following structure:
        {
            "summary": str,
            "actions": [str, ...],
            "weekly_plan": [str, ...],
            "weights": {qid: {theme: int, ...}, ...},
            "deep_insight": str,
            "future_snapshot": str,
            "affirmation": str,
            "quote": str,
            "top_theme_boosters": [str, ...],
            "pitfalls": [str, ...]
        }
        Ensure all strings are concise (max 200 chars per string), safe for PDF rendering, and avoid special characters that might break PDF generation. Avoid long continuous words.
        """
        user_prompt = json.dumps({
            "scores": scores,
            "top3": top3,
            "free_responses": [{k: clean_text(v, max_len=50) if k == "answer" else v for k, v in fr.items()} for fr in free_responses],
            "first_name": clean_text(first_name, max_len=50),
            "horizon_weeks": horizon_weeks
        })
        content, usage, path = _call_openai_json(HIGH_MODEL, system_prompt, user_prompt, FALLBACK_CAP)
        if content:
            try:
                result = json.loads(content)
                # Clean AI-generated strings
                for key in ["summary", "deep_insight", "future_snapshot", "affirmation", "quote"]:
                    if key in result:
                        result[key] = clean_text(result[key], max_len=50)
                for key in ["actions", "weekly_plan", "top_theme_boosters", "pitfalls"]:
                    if key in result:
                        result[key] = [clean_text(item, max_len=50) for item in result[key]]
                return result
            except json.JSONDecodeError:
                st.warning("AI returned invalid JSON. Using default template.")
                return None
        return None
    except Exception as e:
        st.warning(f"AI processing failed: {str(e)}. Using default template.")
        return None

# ----------------------------------------------------------------------------------
# PDF GENERATOR (with error handling)
# ----------------------------------------------------------------------------------
def make_pdf_bytes(
    first_name: str,
    email: str,
    scores: Dict[str, int],
    top3: List[str],
    sections: dict,
    free_responses: List[dict],
    logo_path: Optional[str],
) -> bytes:
    try:
        pdf, unicode_ok = create_pdf_with_unicode()
        pdf.add_page()
        # Logo if available
        if logo_path:
            try:
                pdf.image(logo_path, x=15, y=15, w=30)
            except Exception:
                pass
        # Title
        setf(pdf, unicode_ok, "B", 18)
        mc(pdf, clean_text(REPORT_TITLE, max_len=50), 8, unicode_ok)
        setf(pdf, unicode_ok, "I", 10)
        ts = datetime.datetime.now().strftime("%B %d, %Y")
        mc(pdf, f"Generated for {clean_text(first_name or 'you', max_len=50)} on {ts}", unicode_ok)
        pdf.ln(3)
        setf(pdf, unicode_ok, "", 12)

        # Greet
        greet = f"Hi {clean_text(first_name, max_len=50)}," if first_name else "Hi,"
        mc(pdf, greet, unicode_ok)
        pdf.ln(2)

        # Summary
        if sections.get("summary"):
            mc(pdf, clean_text(sections["summary"], max_len=50), unicode_ok)
            pdf.ln(2)

        # Themes
        setf(pdf, unicode_ok, "B", 14); mc(pdf, "Your Top Themes", unicode_ok)
        setf(pdf, unicode_ok, "", 12)
        for i, theme in enumerate(top3, 1):
            mc(pdf, f"{i}. {clean_text(theme, max_len=50)}", unicode_ok)
        pdf.ln(2)

        # Deep insight
        if sections.get("deep_insight"):
            setf(pdf, unicode_ok, "B", 14); mc(pdf, "Deep Insight", unicode_ok)
            setf(pdf, unicode_ok, "", 12)
            mc(pdf, clean_text(sections["deep_insight"], max_len=50), unicode_ok)
            pdf.ln(2)

        # Future snapshot
        if sections.get("future_snapshot"):
            setf(pdf, unicode_ok, "B", 14); mc(pdf, f"Your Future Snapshot ({sections.get('horizon_weeks', 4)} weeks from now)", unicode_ok)
            setf(pdf, unicode_ok, "", 12)
            mc(pdf, clean_text(sections["future_snapshot"], max_len=50), unicode_ok)
            pdf.ln(2)

        # Actions
        if sections.get("actions"):
            setf(pdf, unicode_ok, "B", 14); mc(pdf, "Tiny Actions to Start Today", unicode_ok)
            setf(pdf, unicode_ok, "", 12)
            for a in sections["actions"]:
                mc(pdf, f"* {clean_text(a, max_len=50)}", unicode_ok)
            pdf.ln(2)

        # Weekly plan
        if sections.get("weekly_plan"):
            setf(pdf, unicode_ok, "B", 14); mc(pdf, "Your Signature Week", unicode_ok)
            setf(pdf, unicode_ok, "", 12)
            for i, step in enumerate(sections["weekly_plan"], 1):
                mc(pdf, f"Day {i}: {clean_text(step, max_len=50)}", unicode_ok)
            pdf.ln(2)

        # Balancing Opportunity
        lows = [name for name, _ in sorted(scores.items(), key=lambda x: x[1])[:2]]
        if lows:
            setf(pdf, unicode_ok, "B", 14); mc(pdf, "Balancing Opportunity", unicode_ok)
            setf(pdf, unicode_ok, "", 12)
            for theme in lows:
                tip = balancing_suggestion(theme)
                mc(pdf, f"{theme}: {clean_text(tip, max_len=50)}", unicode_ok)
            pdf.ln(2)

        # Boosters / pitfalls
        if sections.get("top_theme_boosters") or sections.get("pitfalls"):
            setf(pdf, unicode_ok, "B", 14); mc(pdf, "Amplify what works / Avoid what trips you", unicode_ok)
            if sections.get("top_theme_boosters"):
                setf(pdf, unicode_ok, "B", 12); mc(pdf, "Boosters", unicode_ok)
                setf(pdf, unicode_ok, "", 12)
                for b in sections.get("top_theme_boosters", []):
                    mc(pdf, f"* {clean_text(b, max_len=50)}", unicode_ok)
            if sections.get("pitfalls"):
                setf(pdf, unicode_ok, "B", 12); mc(pdf, "Pitfalls", unicode_ok)
                setf(pdf, unicode_ok, "", 12)
                for p in sections.get("pitfalls", []):
                    mc(pdf, f"- {clean_text(p, max_len=50)}", unicode_ok)
            pdf.ln(2)

        # Affirmation / quote
        if sections.get("affirmation") or sections.get("quote"):
            setf(pdf, unicode_ok, "B", 12); mc(pdf, "Keep this in view", unicode_ok)
            setf(pdf, unicode_ok, "I", 11)
            if sections.get("affirmation"):
                mc(pdf, f"Affirmation: {clean_text(sections['affirmation'], max_len=50)}", unicode_ok)
            if sections.get("quote"):
                qtext = f"\"{clean_text(sections['quote'], max_len=50)}\""
                mc(pdf, qtext, unicode_ok)
            pdf.ln(2)
            setf(pdf, unicode_ok, "", 12)

        # Your words we heard
        if free_responses:
            setf(pdf, unicode_ok, "B", 14); mc(pdf, "Your words we heard", unicode_ok)
            setf(pdf, unicode_ok, "", 12)
            for fr in free_responses:
                if not fr.get("answer"):
                    continue
                mc(pdf, f"* {clean_text(fr.get('question',''), max_len=50)}", unicode_ok)
                mc(pdf, f"  {clean_text(fr.get('answer',''), max_len=50)}", unicode_ok)
                pdf.ln(2)

        # Page hint before checklist
        pdf.ln(3)
        setf(pdf, unicode_ok, "B", 12)
        mc(pdf, "On the next page: a printable 'Signature Week — At a glance' checklist you can use right away.", unicode_ok)

        # Checklist page
        pdf.add_page()
        setf(pdf, unicode_ok, "B", 16)
        mc(pdf, "Signature Week — At a glance", unicode_ok)
        setf(pdf, unicode_ok, "", 12)
        mc(pdf, "A simple plan you can print or screenshot. Check items off as you go.", unicode_ok)
        pdf.ln(2)

        week_items = sections.get("weekly_plan") or []
        if not week_items:
            week_items = [f"Do one small action for {t}" for t in top3] + ["Reflect and set next step"]
        for i, item in enumerate(week_items[:7]):
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.rect(x, y + 1.5, 4, 4)
            pdf.set_x(x + 6)
            mc(pdf, f"Day {i+1}: {clean_text(item, max_len=50)}", unicode_ok)

        pdf.ln(2)
        setf(pdf, unicode_ok, "B", 14); mc(pdf, "Tiny Progress Tracker", unicode_ok)
        setf(pdf, unicode_ok, "", 12)
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
            mc(pdf, clean_text(m, max_len=50), unicode_ok)
        pdf.ln(2)

        setf(pdf, unicode_ok, "I", 10); pdf.ln(2)
        mc(pdf, "Life Minus Work • This report is a starting point for reflection. Nothing here is medical or financial advice.", unicode_ok)
        setf(pdf, unicode_ok, "", 12)

        # Output
        raw = pdf.output(dest="S")
        return raw if isinstance(raw, bytes) else raw.encode("latin-1", errors="replace")
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}. Please try shorter inputs or fewer special characters.")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "", 12)
        pdf.set_margins(left=15, top=15, right=15)
        pdf.multi_cell(0, 10, "Error generating report. Please try again with shorter text or fewer special characters.")
        return pdf.output(dest="S")  # Always return bytes

# ----------------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------------
st.title(APP_TITLE)
st.write("Answer 15 questions, add your own reflections, and instantly download a personalized PDF summary.")

# First name
if "first_name" not in st.session_state:
    st.session_state["first_name"] = ""
first_name = st.text_input("First name", st.session_state["first_name"], max_chars=50)
if first_name:
    st.session_state["first_name"] = clean_text(first_name.strip(), max_len=50)

# Load questions
questions, _ = load_questions()

answers: Dict[str, dict] = {}
free_responses: List[dict] = []

with st.expander("Personalization options"):
    horizon_weeks = st.slider("Future snapshot horizon (weeks)", 2, 8, 4)

# Render questions
for q in questions:
    st.subheader(q["text"])
    options = [c["label"] for c in q["choices"]] + ["✍️ I'll write my own answer"]
    selected = st.radio("Choose one:", options, index=None, key=f"{q['id']}_choice")
    choice_idx = None
    free_text_val = None

    if selected == "✍️ I'll write my own answer":
        free_text_val = st.text_area("Your answer", key=f"{q['id']}_free", height=80, placeholder="Type your own response...", max_chars=1000)
        if free_text_val:
            free_text_val = clean_text(free_text_val, max_len=50)
    elif selected is not None:
        idx = options.index(selected)
        if idx < len(q["choices"]):
            choice_idx = idx

    answers[q["id"]] = {"choice_idx": choice_idx, "free_text": free_text_val}
    if free_text_val:
        free_responses.append({"id": q["id"], "question": q["text"], "answer": free_text_val})
    st.divider()

# Email + consent form
st.subheader("Email & Download")
with st.form("finish_form"):
    email_val = st.text_input("Your email (for your download link)", key="email_input", placeholder="you@example.com")
    consent_val = st.checkbox(
        "I agree to receive my results and occasional updates from Life Minus Work.",
        key="consent_input",
        value=st.session_state.get("consent_input", False),
    )
    submit_clicked = st.form_submit_button("Generate My Personalized Report")
    if submit_clicked:
        if not email_val or not consent_val:
            st.error("Please enter your email and give consent to continue.")
        else:
            st.session_state["email"] = email_val.strip()
            st.session_state["consent"] = True
            st.session_state["request_report"] = True
            st.toast("Generating your report…", icon="⏳")

# Run generation
if st.session_state.get("request_report"):
    st.session_state["request_report"] = False

    scores = compute_scores(answers, questions)
    top3 = top_themes(scores, 3)

    # AI generation
    sections = {"summary": "", "actions": [], "weekly_plan": [], "weights": {}}
    if USE_AI:
        maybe = ai_sections_and_weights(
            scores,
            top3,
            free_responses,
            st.session_state.get("first_name", ""),
            horizon_weeks=horizon_weeks,
        )
        dbg = st.session_state.get("ai_debug") or {}
        tok = st.session_state.get("token_usage") or {}
        with st.expander("AI generation details (debug)", expanded=False):
            if dbg:
                for k, v in dbg.items():
                    if k == "raw_head" and isinstance(v, str):
                        st.text_area("raw_head (first 800 chars)", v, height=200)
                    else:
                        st.write(f"{k}: {v}")
            p = Path("/tmp/last_ai.json")
            if p.exists():
                st.download_button(
                    "Download last_ai.json",
                    data=p.read_bytes(),
                    file_name="last_ai.json",
                    mime="application/json",
                )
        with st.expander("Token usage (one run)", expanded=False):
            if tok:
                st.write(
                    f"Model: {tok.get('model','?')} | path: {tok.get('path','?')} | cap_used: {tok.get('cap_used','?')}"
                )
                st.write(
                    f"Input tokens: {tok.get('input','?')} | Output tokens: {tok.get('output','?')} | Total: {tok.get('total','?')}"
                )
                st.caption(f"Timestamp: {tok.get('ts','?')}")
            else:
                st.write("No usage returned by the API (some paths/models omit it).")

        if maybe:
            sections.update(maybe)
            if sections.get("weights"):
                scores = apply_free_text_weights(scores, sections["weights"])
            sections["horizon_weeks"] = horizon_weeks
        else:
            st.warning("AI could not generate JSON this run — using a concise template instead.")

    # Fallback content if AI missing
    if not sections.get("deep_insight"):
        top1 = top3[0] if top3 else "what energizes you"
        sections.update({
            "deep_insight": f"Thank you for completing the Reflection Quiz, {clean_text(st.session_state.get('first_name','Friend'), max_len=50)}.",
            "actions": [
                "Choose one tiny step you can take this week.",
                "Tell a friend your plan—gentle accountability.",
                "Schedule 20 minutes for reflection or journaling.",
            ],
            "weekly_plan": [
                "Name your intention.",
                "15–20 minutes of learning or practice.",
                "Reach out to someone who energizes you.",
                "Take a calm walk or mindful pause.",
                "Do one small adventurous thing.",
                "Offer help or encouragement to someone.",
                "Review your week and set the next tiny step.",
            ],
            "future_snapshot": (
                f"It is {horizon_weeks} weeks later. You have stayed close to what matters, "
                f"protecting time for {top1}. A few tiny actions, repeated, build confidence. "
                "You pause, adjust, and keep going."
            ),
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
            "from_words": {},
            "micro_pledge": "",
        })

    logo_path = get_logo_png_path()
    pdf_bytes = make_pdf_bytes(
        st.session_state.get("first_name", ""),
        st.session_state.get("email", ""),
        scores,
        top3,
        sections,
        free_responses,
        logo_path,
    )

    if isinstance(pdf_bytes, bytes):
        st.success("Your personalized report is ready!")
        st.download_button(
            "📥 Download Your PDF Report",
            data=pdf_bytes,
            file_name="LifeMinusWork_Reflection_Report.pdf",
            mime="application/pdf",
        )
    else:
        st.error("Failed to generate a valid PDF. Please try again with shorter inputs.")

    # (Optional) CSV logging to /tmp
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
                clean_text(st.session_state.get("first_name", ""), max_len=50),
                st.session_state.get("email", ""),
                json.dumps(scores),
                json.dumps(top3),
            ])
        st.caption("Saved to /tmp/responses.csv (Cloud-safe, ephemeral).")
    except Exception as e:
        st.caption(f"Could not save responses (demo only). {e}")

# Convenience “Test OpenAI” control (optional)
with st.expander("AI status (debug)", expanded=False):
    st.write("AI enabled:", USE_AI)
    st.write("Model:", HIGH_MODEL)
    st.write("Max tokens:", MAX_TOK_HIGH, "(fallback", FALLBACK_CAP, ")")
    if USE_AI and st.button("Test OpenAI now"):
        try:
            raw, usage, path = _call_openai_json(
                HIGH_MODEL,
                "Return strict JSON only.",
                'Return {"ok": true} as JSON.',
                cap=128,
            )
            msg = f"OK — via {path}. Output: {raw}"
            if usage:
                msg += f" | usage: in={usage.get('input')} out={usage.get('output')} total={usage.get('total')}"
            st.success(msg)
        except Exception as e:
            st.error(f"OpenAI error: {e}")
