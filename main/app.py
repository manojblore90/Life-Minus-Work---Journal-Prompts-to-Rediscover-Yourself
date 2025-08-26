# app.py — Life Minus Work (roll-back with AI working + PDF hardening)
# -------------------------------------------------------------------
import os, sys, re, json, unicodedata, datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
from PIL import Image
from fpdf import FPDF

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------------
# CONFIG & PAGE
# -----------------------------
APP_TITLE    = "Life Minus Work — Reflection Quiz (15 questions)"
REPORT_TITLE = "Your Reflection Report"
THEMES       = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="centered")

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
# Keep the same model path you were using when AI worked well:
HIGH_MODEL = get_secret("OPENAI_HIGH_MODEL", "gpt-5-mini")
MAX_TOK_HIGH = int(get_secret("MAX_OUTPUT_TOKENS_HIGH", "8000"))
FALLBACK_CAP = int(get_secret("MAX_OUTPUT_TOKENS_FALLBACK", "6000"))

# -----------------------------
# TEMP DIAGNOSTICS
# -----------------------------
with st.expander("🔧 Diagnostics (temporary)", expanded=False):
    st.write("Python:", sys.version.split()[0])
    here = Path(__file__).parent
    st.write("__file__:", __file__)
    st.write("cwd:", os.getcwd())
    try:
        st.write("Files near app.py:", [p.name for p in here.iterdir()])
    except Exception as e:
        st.write("Dir list failed:", e)
    try:
        import fpdf as _fp
        st.write("fpdf2 version:", getattr(_fp, "__version__", "unknown"))
    except Exception:
        pass
    masked = (OPENAI_API_KEY[:4] + "…" + OPENAI_API_KEY[-4:]) if OPENAI_API_KEY else "None"
    st.write("OPENAI_API_KEY present:", bool(OPENAI_API_KEY), "| key:", masked if OPENAI_API_KEY else "—")
    st.write("Model:", HIGH_MODEL, "| MAX_TOK_HIGH:", MAX_TOK_HIGH, "| FALLBACK_CAP:", FALLBACK_CAP)

# -----------------------------
# TEXT CLEANING
# -----------------------------
def _ascii_only(s: str) -> str:
    return (s.replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"')
             .replace("–", "-").replace("—", "-")
             .replace("…", "...").replace("•", "*"))

def clean_text(s: str, max_len: int = 1000, ascii_fallback: bool = False) -> str:
    if not s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    # Guard against ultra-long tokens
    parts = []
    for t in s.split():
        parts.append(t if len(t) <= max_len else t[:max_len] + "...")
    s = " ".join(parts)
    if ascii_fallback:
        s = _ascii_only(s)
    return s

# -----------------------------
# WIDTH-SAFE MultiCell (fixes crash)
# -----------------------------
def mc(pdf: "FPDF", text: str, h: float = 6, unicode_ok: bool = True):
    """
    Crash-proof MultiCell:
      - explicit width (no width=0)
      - normalizes text
      - retries with ASCII
      - final fallback prints safe notice (never throws)
    """
    try:
        w = float(pdf.w) - float(pdf.l_margin) - float(pdf.r_margin)
    except Exception:
        w = 180.0
    if w <= 0:
        w = 180.0

    s = clean_text((text or "").replace("\r\n", "\n").replace("\r", "\n"),
                   ascii_fallback=not unicode_ok)

    # Try 1: full text
    try:
        pdf.multi_cell(w, h, s)
        return
    except Exception:
        pass

    # Try 2: ASCII fallback
    try:
        s2 = clean_text(s, ascii_fallback=True)
        pdf.multi_cell(w, h, s2)
        return
    except Exception:
        pass

    # Try 3: reset to core font + safe line
    try:
        pdf.set_font("Helvetica", "", 12)
    except Exception:
        pass
    try:
        pdf.multi_cell(w, h, "[...content truncated...]")
    except Exception:
        return

# -----------------------------
# BYTES SAFETY
# -----------------------------
def to_bytes(x: Any) -> bytes:
    if x is None:
        return b""
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, str):
        return x.encode("latin-1", errors="replace")
    if hasattr(x, "read"):
        try:
            data = x.read()
            return data if isinstance(data, bytes) else bytes(str(data), "utf-8", "ignore")
        except Exception:
            return b""
    try:
        return bytes(x)
    except Exception:
        return bytes(str(x), "utf-8", "ignore")

# -----------------------------
# UNICODE PDF SUPPORT
# -----------------------------
def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return str(p)
    return None

def create_pdf_with_unicode() -> Tuple[FPDF, bool]:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    here = Path(__file__).parent
    candidates_regular = [
        here / "DejaVuSans.ttf",
        here / "assets" / "DejaVuSans.ttf",
        here / "assets" / "fonts" / "DejaVuSans.ttf",
        here / "NotoSans-Regular.ttf",
        here / "assets" / "NotoSans-Regular.ttf",
        here / "assets" / "fonts" / "NotoSans-Regular.ttf",
    ]
    candidates_bold = [
        here / "DejaVuSans-Bold.ttf",
        here / "assets" / "DejaVuSans-Bold.ttf",
        here / "assets" / "fonts" / "DejaVuSans-Bold.ttf",
        here / "NotoSans-Bold.ttf",
        here / "assets" / "NotoSans-Bold.ttf",
        here / "assets" / "fonts" / "NotoSans-Bold.ttf",
    ]
    candidates_italic = [
        here / "DejaVuSans-Oblique.ttf",
        here / "assets" / "DejaVuSans-Oblique.ttf",
        here / "assets" / "fonts" / "DejaVuSans-Oblique.ttf",
        here / "NotoSans-Italic.ttf",
        here / "assets" / "NotoSans-Italic.ttf",
        here / "assets" / "fonts" / "NotoSans-Italic.ttf",
    ]
    reg = _first_existing(candidates_regular)
    if reg:
        try:
            pdf.add_font("LMW", "", reg, uni=True)
            b = _first_existing(candidates_bold)
            if b:
                try: pdf.add_font("LMW", "B", b, uni=True)
                except Exception: pass
            i = _first_existing(candidates_italic)
            if i:
                try: pdf.add_font("LMW", "I", i, uni=True)
                except Exception: pass
            pdf.set_font("LMW", "", 12)
            return pdf, True
        except Exception:
            pass

    # Fallback Latin-1 (still safe with ASCII)
    pdf.set_font("Helvetica", "", 12)
    return pdf, False

def setf(pdf: FPDF, unicode_ok: bool, style: str = "", size: int = 12):
    if unicode_ok:
        try:
            pdf.set_font("LMW", style or "", size)
        except Exception:
            pdf.set_font("LMW", "", size)
    else:
        pdf.set_font("Helvetica", style or "", size)

# -----------------------------
# LOGO
# -----------------------------
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

# -----------------------------
# QUESTIONS
# -----------------------------
def load_questions(filename="questions.json"):
    base_dir = Path(__file__).parent
    path = base_dir / filename
    if not path.exists():
        st.error(f"Could not find {filename} at {path}. Make sure it's next to app.py.")
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

# -----------------------------
# SCORING
# -----------------------------
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

def top_themes(scores: Dict[str, int], k: int = 3) -> List[str]:
    return [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

def balancing_suggestion(theme: str) -> str:
    suggestions = {
        "Identity": "Choose one tiny ritual that reflects who you are becoming.",
        "Growth": "Pick one skill and practice 15 minutes today.",
        "Connection": "Send a 3-line check-in to someone who matters.",
        "Peace": "Name a 10-minute wind-down you will repeat daily.",
        "Adventure": "Plan a 30–60 minute micro-adventure within 7 days.",
        "Contribution": "Offer one concrete act of help this week.",
    }
    return suggestions.get(theme, "Take one small, visible step this week.")

# -----------------------------
# OPENAI (JSON mode + token tracker)
# -----------------------------
def _call_openai_json(model: str, system: str, user: str, cap: int):
    if not (USE_AI and OpenAI):
        raise RuntimeError("OpenAI not configured")
    client = OpenAI()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=cap,               # IMPORTANT for latest SDK/models
        response_format={"type": "json_object"}, # Strict JSON
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

def ai_sections_and_weights(scores, top3, free_responses, first_name, horizon_weeks=4) -> Optional[dict]:
    if not USE_AI:
        return None
    st.session_state["ai_debug"] = {}
    try:
        packed, allowed_ids = [], []
        for i, fr in enumerate(free_responses or []):
            if not isinstance(fr, dict):
                continue
            ans = str(fr.get("answer", "")).strip()
            if not ans:
                continue
            qid = str(fr.get("id") or fr.get("qid") or f"free_{i+1}")
            qtxt = str(fr.get("question", "")).strip()[:160]
            allowed_ids.append(qid)
            packed.append({"id": qid, "q": qtxt, "a": ans[:280]})

        score_lines = ", ".join([f"{k}: {v}" for k, v in scores.items()])

        prompt = (
            "You are a warm, practical life coach. Return ONLY valid JSON with keys:\n"
            "  archetype, core_need,\n"
            "  deep_insight (400–600 words, address user by first name),\n"
            "  why_now (120–180 words),\n"
            "  strengths (4–6), energizers (4), drainers (4), tensions (2–3), blindspot,\n"
            "  actions (EXACTLY 3), if_then (EXACTLY 3), weekly_plan (7),\n"
            "  affirmation (<=15 words), quote (<=20 words),\n"
            "  signature_metaphor (<=12 words), signature_sentence (<=20 words),\n"
            "  top_theme_boosters (<=4), pitfalls (<=4),\n"
            "  future_snapshot (150–220 words, second-person, present tense, as if {h} weeks later),\n"
            "  from_words { themes(3), quotes(2–3, <=12 words each), insight(80–120 words), ritual, relationship_moment, stress_reset },\n"
            "  micro_pledge (first-person <=28 words),\n"
            "  weights (question_id -> {theme:int in [-2,2]}).\n"
            f"User first name: {first_name or 'Friend'}.\n"
            f"Theme scores: {score_lines}.\n"
            f"Top 3 themes: {', '.join(top3)}.\n"
            f"Horizon weeks: {horizon_weeks}.\n"
            f"Free-text answers: {json.dumps(packed, ensure_ascii=False)}\n"
            "IMPORTANT: Only use question_id keys for 'weights' from this list:\n"
            f"{json.dumps(allowed_ids, ensure_ascii=False)}\n"
            "Tone: empathetic, encouraging, plain language. No medical claims. JSON only."
        ).format(h=horizon_weeks)

        system = "Reply with helpful coaching guidance as STRICT JSON only."
        tries = [MAX_TOK_HIGH, 6000, 4000, FALLBACK_CAP, 2500, 1200]

        last_err = None
        for cap in tries:
            try:
                raw, usage, path = _call_openai_json(HIGH_MODEL, system, prompt, cap)
                raw = (raw or "").strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
                    raw = re.sub(r"\n```$", "", raw)

                data = None
                try:
                    data = json.loads(raw)
                except Exception:
                    if "{" in raw and "}" in raw:
                        raw2 = raw[raw.find("{"): raw.rfind("}") + 1]
                        data = json.loads(raw2)

                if not isinstance(data, dict):
                    raise ValueError("No JSON object found in completion.")

                st.session_state["ai_debug"] = {
                    "path": path, "cap_used": cap, "raw_head": raw[:800], "raw_len": len(raw),
                }
                if usage:
                    st.session_state["token_usage"] = {
                        "model": HIGH_MODEL, "path": path, "cap_used": cap,
                        "input": usage.get("input"), "output": usage.get("output"),
                        "total": usage.get("total"),
                        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    }
                try:
                    Path("/tmp/last_ai.json").write_text(raw, encoding="utf-8")
                except Exception:
                    pass

                sg = lambda k: str(data.get(k, "") or "")
                out = {
                    "archetype": sg("archetype"),
                    "core_need": sg("core_need"),
                    "deep_insight": sg("deep_insight"),
                    "why_now": sg("why_now"),
                    "strengths": [str(x) for x in (data.get("strengths") or [])][:6],
                    "energizers": [str(x) for x in (data.get("energizers") or [])][:4],
                    "drainers": [str(x) for x in (data.get("drainers") or [])][:4],
                    "tensions": [str(x) for x in (data.get("tensions") or [])][:3],
                    "blindspot": sg("blindspot"),
                    "actions": [str(x) for x in (data.get("actions") or [])][:3],
                    "if_then": [str(x) for x in (data.get("if_then") or [])][:3],
                    "weekly_plan": [str(x) for x in (data.get("weekly_plan") or [])][:7],
                    "affirmation": sg("affirmation"),
                    "quote": sg("quote"),
                    "signature_metaphor": sg("signature_metaphor"),
                    "signature_sentence": sg("signature_sentence"),
                    "top_theme_boosters": [str(x) for x in (data.get("top_theme_boosters") or [])][:4],
                    "pitfalls": [str(x) for x in (data.get("pitfalls") or [])][:4],
                    "future_snapshot": sg("future_snapshot"),
                    "from_words": {},
                    "micro_pledge": sg("micro_pledge"),
                    "weights": {},
                }
                fw = data.get("from_words") or {}
                if isinstance(fw, dict):
                    out["from_words"] = {
                        "themes": [str(x) for x in (fw.get("themes") or [])][:3],
                        "quotes": [str(x) for x in (fw.get("quotes") or [])][:3],
                        "insight": str(fw.get("insight", "")),
                        "ritual": str(fw.get("ritual", "")),
                        "relationship_moment": str(fw.get("relationship_moment", "")),
                        "stress_reset": str(fw.get("stress_reset", "")),
                    }
                weights = data.get("weights") or {}
                if isinstance(weights, dict):
                    clean_w = {}
                    for qid, w in weights.items():
                        if not isinstance(w, dict):
                            continue
                        m = {}
                        for theme, val in w.items():
                            if theme in THEMES:
                                try:
                                    iv = int(val)
                                    iv = max(-2, min(2, iv))
                                    m[theme] = iv
                                except Exception:
                                    pass
                        if m:
                            clean_w[str(qid)] = m
                    out["weights"] = clean_w
                return out
            except Exception as e:
                last_err = e
                continue
        st.session_state["ai_debug"] = {"error": f"{type(last_err).__name__}: {last_err}"}
        return None
    except Exception as e:
        st.session_state["ai_debug"] = {"fatal": f"{type(e).__name__}: {e}"}
        return None

# -----------------------------
# PDF HELPERS
# -----------------------------
def draw_scores_barchart(pdf: FPDF, unicode_ok: bool, scores: Dict[str, int]):
    setf(pdf, unicode_ok, "B", 14)
    mc(pdf, "Your Theme Snapshot", unicode_ok=unicode_ok)
    setf(pdf, unicode_ok, "", 12)
    max_score = max(max(scores.values()), 1)
    bar_w_max = 120
    x_left = pdf.get_x() + 10
    y = pdf.get_y()
    for theme in THEMES:
        val = scores.get(theme, 0)
        bar_w = (val / max_score) * bar_w_max
        pdf.set_xy(x_left, y)
        pdf.cell(35, 6, _ascii_only(theme))
        pdf.set_fill_color(30, 144, 255)
        pdf.rect(x_left + 38, y + 1.3, bar_w, 4.5, "F")
        pdf.set_xy(x_left + 38 + bar_w + 2, y)
        pdf.cell(0, 6, _ascii_only(str(val)))
        y += 7
    pdf.set_y(y + 4)

def paragraph(pdf: FPDF, unicode_ok: bool, title: str, body: str):
    setf(pdf, unicode_ok, "B", 14)
    mc(pdf, title, unicode_ok=unicode_ok)
    setf(pdf, unicode_ok, "", 12)
    for line in str(body).split("\n"):
        mc(pdf, line, unicode_ok=unicode_ok)
    pdf.ln(2)

def checkbox_line(pdf: FPDF, unicode_ok: bool, text: str):
    x = pdf.get_x(); y = pdf.get_y()
    pdf.rect(x, y + 1.5, 4, 4)
    pdf.set_x(x + 6)
    mc(pdf, text, unicode_ok=unicode_ok)

def label_value(pdf: FPDF, unicode_ok: bool, label: str, value: str):
    setf(pdf, unicode_ok, "B", 12); mc(pdf, label, unicode_ok=unicode_ok)
    setf(pdf, unicode_ok, "", 12);  mc(pdf, value, unicode_ok=unicode_ok)

def future_callout(pdf: FPDF, unicode_ok: bool, weeks: int, text: str):
    pdf.set_text_color(30, 60, 120)
    setf(pdf, unicode_ok, "B", 14)
    mc(pdf, f"Future Snapshot — {weeks} weeks", unicode_ok=unicode_ok)
    pdf.set_text_color(0, 0, 0)
    setf(pdf, unicode_ok, "I", 12)
    mc(pdf, text, unicode_ok=unicode_ok)
    pdf.ln(2)
    setf(pdf, unicode_ok, "", 12)

def left_bar_callout(pdf: FPDF, unicode_ok: bool, title: str, body: str, bullets=None):
    if bullets is None:
        bullets = []
    x = pdf.get_x(); y = pdf.get_y()
    pdf.set_fill_color(30, 144, 255)
    pdf.rect(x, y, 2, 6, "F")
    pdf.set_x(x + 4)
    setf(pdf, unicode_ok, "B", 13)
    mc(pdf, title, unicode_ok=unicode_ok)
    pdf.set_x(x + 4)
    setf(pdf, unicode_ok, "", 12)
    mc(pdf, body, unicode_ok=unicode_ok)
    for b in bullets:
        pdf.set_x(x + 4)
        pdf.cell(4, 6, "*")
        mc(pdf, b, unicode_ok=unicode_ok)
    pdf.ln(1)

# -----------------------------
# PDF MAKE
# -----------------------------
def make_pdf_bytes(first_name: str, email: str, scores: Dict[str,int], top3: List[str],
                   sections: dict, free_responses: List[dict], logo_path: Optional[str]) -> bytes:
    pdf, unicode_ok = create_pdf_with_unicode()
    pdf.add_page()

    # IMPORTANT: set a font BEFORE first mc()
    setf(pdf, unicode_ok, "B", 18)

    # Logo
    if logo_path:
        try:
            pdf.image(logo_path, w=40); pdf.ln(2)
        except Exception:
            pass

    mc(pdf, REPORT_TITLE, unicode_ok=unicode_ok)
    setf(pdf, unicode_ok, "", 12)
    today = datetime.date.today().strftime("%d %b %Y")
    greet = f"Hi {first_name}," if first_name else "Hello,"
    mc(pdf, greet, unicode_ok=unicode_ok)
    mc(pdf, f"Date: {today}", unicode_ok=unicode_ok)
    if email:
        mc(pdf, f"Email: {email}", unicode_ok=unicode_ok)
    pdf.ln(1)

    if sections.get("archetype") or sections.get("core_need"):
        label_value(pdf, unicode_ok, "Archetype", sections.get("archetype","") or "—")
        label_value(pdf, unicode_ok, "Core Need", sections.get("core_need","") or "—")
        if sections.get("signature_metaphor"):
            label_value(pdf, unicode_ok, "Signature Metaphor", sections.get("signature_metaphor",""))
        if sections.get("signature_sentence"):
            label_value(pdf, unicode_ok, "Signature Sentence", sections.get("signature_sentence",""))
        pdf.ln(1)

    setf(pdf, unicode_ok, "B", 14); mc(pdf, "Top Themes", unicode_ok=unicode_ok)
    setf(pdf, unicode_ok, "", 12);  mc(pdf, ", ".join(top3), unicode_ok=unicode_ok); pdf.ln(1)

    draw_scores_barchart(pdf, unicode_ok, scores)

    fw = sections.get("from_words") or {}
    if isinstance(fw, dict) and (fw.get("insight") or fw.get("themes") or fw.get("quotes")):
        quotes = [f'"{q}"' for q in fw.get("quotes", []) if q]
        left_bar_callout(pdf, unicode_ok, "From your words", fw.get("insight",""), bullets=quotes)
        keep = [("Daily ritual", fw.get("ritual","")),
                ("Connection moment", fw.get("relationship_moment","")),
                ("Stress reset", fw.get("stress_reset",""))]
        if any(v for _, v in keep):
            setf(pdf, unicode_ok, "B", 12); mc(pdf, "One-liners to keep", unicode_ok=unicode_ok)
            setf(pdf, unicode_ok, "", 12)
            for lbl, val in keep:
                if val: mc(pdf, f"{lbl}: {val}", unicode_ok=unicode_ok)
            pdf.ln(1)
    if sections.get("micro_pledge"):
        label_value(pdf, unicode_ok, "Personal pledge", sections["micro_pledge"]); pdf.ln(1)

    if sections.get("deep_insight"):
        paragraph(pdf, unicode_ok, "What this really says about you", sections["deep_insight"])
    if sections.get("why_now"):
        label_value(pdf, unicode_ok, "Why this matters now", sections["why_now"]); pdf.ln(1)

    if sections.get("future_snapshot"):
        future_callout(pdf, unicode_ok, sections.get("horizon_weeks", 4), sections["future_snapshot"])

    if sections.get("strengths"):
        setf(pdf, unicode_ok, "B", 14); mc(pdf, "Signature strengths", unicode_ok=unicode_ok)
        setf(pdf, unicode_ok, "", 12)
        for s in sections["strengths"]:
            pdf.cell(4, 6, "*"); mc(pdf, s, unicode_ok=unicode_ok)
        pdf.ln(1)

    if sections.get("energizers") or sections.get("drainers"):
        setf(pdf, unicode_ok, "B", 14); mc(pdf, "Energy map", unicode_ok=unicode_ok)
        setf(pdf, unicode_ok, "B", 12); mc(pdf, "Energizers", unicode_ok=unicode_ok)
        setf(pdf, unicode_ok, "", 12)
        for e in sections.get("energizers", []):
            pdf.cell(4, 6, "+"); mc(pdf, e, unicode_ok=unicode_ok)
        pdf.ln(1)
        setf(pdf, unicode_ok, "B", 12); mc(pdf, "Drainers", unicode_ok=unicode_ok)
        setf(pdf, unicode_ok, "", 12)
        for d in sections.get("drainers", []):
            pdf.cell(4, 6, "-"); mc(pdf, d, unicode_ok=unicode_ok)
        pdf.ln(1)

    if sections.get("tensions"):
        setf(pdf, unicode_ok, "B", 14); mc(pdf, "Hidden tensions", unicode_ok=unicode_ok)
        setf(pdf, unicode_ok, "", 12)
        for t in sections["tensions"]:
            pdf.cell(4, 6, "*"); mc(pdf, t, unicode_ok=unicode_ok)
        pdf.ln(1)
    if sections.get("blindspot"):
        label_value(pdf, unicode_ok, "Watch-out (gentle blind spot)", sections["blindspot"]); pdf.ln(1)

    if sections.get("actions"):
        setf(pdf, unicode_ok, "B", 14); mc(pdf, "3 next-step actions (7 days)", unicode_ok=unicode_ok)
        setf(pdf, unicode_ok, "", 12)
        for a in sections["actions"]:
            checkbox_line(pdf, unicode_ok, a)
        pdf.ln(1)

    if sections.get("if_then"):
        setf(pdf, unicode_ok, "B", 14); mc(pdf, "Implementation intentions (If–Then)", unicode_ok=unicode_ok)
        setf(pdf, unicode_ok, "", 12)
        for it in sections.get("if_then", []):
            pdf.cell(4, 6, "*"); mc(pdf, it, unicode_ok=unicode_ok)
        pdf.ln(1)

    if sections.get("weekly_plan"):
        setf(pdf, unicode_ok, "B", 14); mc(pdf, "1-week gentle plan", unicode_ok=unicode_ok)
        setf(pdf, unicode_ok, "", 12)
        for i, item in enumerate(sections["weekly_plan"][:7]):
            mc(pdf, f"Day {i+1}: {item}", unicode_ok=unicode_ok)
        pdf.ln(1)

    lows = [name for name, _ in sorted(scores.items(), key=lambda x: x[1])[:2]]
    if lows:
        setf(pdf, unicode_ok, "B", 14); mc(pdf, "Balancing Opportunity", unicode_ok=unicode_ok)
        setf(pdf, unicode_ok, "", 12)
        for theme in lows:
            tip = balancing_suggestion(theme)
            mc(pdf, f"{theme}: {tip}", unicode_ok=unicode_ok)
        pdf.ln(1)

    if sections.get("affirmation") or sections.get("quote"):
        setf(pdf, unicode_ok, "B", 12); mc(pdf, "Keep this in view", unicode_ok=unicode_ok)
        setf(pdf, unicode_ok, "I", 11)
        if sections.get("affirmation"):
            mc(pdf, f"Affirmation: {sections['affirmation']}", unicode_ok=unicode_ok)
        if sections.get("quote"):
            qtext = f"\"{sections['quote']}\""
            mc(pdf, qtext, unicode_ok=unicode_ok)
        pdf.ln(2)
        setf(pdf, unicode_ok, "", 12)

    if free_responses:
        setf(pdf, unicode_ok, "B", 14); mc(pdf, "Your words we heard", unicode_ok=unicode_ok)
        setf(pdf, unicode_ok, "", 12)
        for fr in free_responses:
            if not fr.get("answer"): continue
            mc(pdf, f"* {fr.get('question','')}", unicode_ok=unicode_ok)
            mc(pdf, f"  {fr.get('answer','')}", unicode_ok=unicode_ok)
            pdf.ln(1)

    pdf.ln(3)
    setf(pdf, unicode_ok, "B", 12)
    mc(pdf, "On the next page: a printable 'Signature Week — At a glance' checklist you can use right away.", unicode_ok=unicode_ok)

    pdf.add_page()
    setf(pdf, unicode_ok, "B", 16)
    mc(pdf, "Signature Week — At a glance", unicode_ok=unicode_ok)
    setf(pdf, unicode_ok, "", 12)
    mc(pdf, "A simple plan you can print or screenshot. Check items off as you go.", unicode_ok=unicode_ok)
    pdf.ln(2)

    week_items = sections.get("weekly_plan") or []
    if not week_items:
        week_items = [f"Do one small action for {t}" for t in top3] + ["Reflect and set next step"]
    for i, item in enumerate(week_items[:7]):
        x = pdf.get_x(); y = pdf.get_y()
        pdf.rect(x, y + 1.5, 4, 4)
        pdf.set_x(x + 6)
        mc(pdf, f"Day {i+1}: {item}", unicode_ok=unicode_ok)

    pdf.ln(2)
    setf(pdf, unicode_ok, "B", 14); mc(pdf, "Tiny Progress Tracker", unicode_ok=unicode_ok)
    setf(pdf, unicode_ok, "", 12)
    milestones = sections.get("actions") or [
        "Choose one tiny step and schedule it.",
        "Tell a friend your plan for gentle accountability.",
        "Spend 20 minutes on your step and celebrate completion."
    ]
    for m in milestones[:3]:
        x = pdf.get_x(); y = pdf.get_y()
        pdf.rect(x, y + 1.5, 4, 4)
        pdf.set_x(x + 6)
        mc(pdf, m, unicode_ok=unicode_ok)
    pdf.ln(2)

    setf(pdf, unicode_ok, "I", 10); pdf.ln(2)
    mc(pdf, "Life Minus Work • This report is a starting point for reflection. Nothing here is medical or financial advice.", unicode_ok=unicode_ok)
    setf(pdf, unicode_ok, "", 12)

    raw = pdf.output(dest="S")
    return raw if isinstance(raw, bytes) else str(raw).encode("latin-1", errors="replace")

# -----------------------------
# UI FLOW
# -----------------------------
st.title(APP_TITLE)
st.write("Answer 15 questions, add your own reflections, and instantly download a personalized PDF summary.")

# First name
if "first_name" not in st.session_state:
    st.session_state["first_name"] = ""
first_name = st.text_input("First name", st.session_state["first_name"])
if first_name:
    st.session_state["first_name"] = first_name.strip()

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
        free_text_val = st.text_area("Your answer", key=f"{q['id']}_free", height=80, placeholder="Type your own response...")
    elif selected is not None:
        idx = options.index(selected)
        if idx < len(q["choices"]):
            choice_idx = idx

    answers[q["id"]] = {"choice_idx": choice_idx, "free_text": free_text_val}
    if free_text_val:
        free_responses.append({"id": q["id"], "question": q["text"], "answer": free_text_val})
    st.divider()

# Email + consent
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

# Generate
if st.session_state.get("request_report"):
    st.session_state["request_report"] = False

    scores = compute_scores(answers, questions)
    top3 = top_themes(scores, 3)

    sections = {"weekly_plan": [], "actions": [], "from_words": {}, "weights": {}}
    if USE_AI:
        maybe = ai_sections_and_weights(
            scores, top3, free_responses, st.session_state.get("first_name", ""), horizon_weeks=horizon_weeks
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

        with st.expander("Token usage (one run)", expanded=True):
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
            # fold weights into scores (only those the model returned)
            weights = sections.get("weights") or {}
            for qid, wmap in weights.items():
                for theme, delta in wmap.items():
                    scores[theme] = scores.get(theme, 0) + int(delta)
            sections["horizon_weeks"] = horizon_weeks
        else:
            st.warning("AI could not generate JSON this run — using a concise template instead.")

    # Minimal fallback if AI missing
    if not sections.get("deep_insight"):
        top1 = top3[0] if top3 else "what energizes you"
        sections.update({
            "deep_insight": f"Thank you for completing the Reflection Quiz, {st.session_state.get('first_name','Friend')}.",
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
    safe_bytes = to_bytes(pdf_bytes)

    st.success("Your personalized report is ready!")
    st.download_button(
        "📥 Download Your PDF Report",
        data=safe_bytes,
        file_name="LifeMinusWork_Reflection_Report.pdf",
        mime="application/pdf",
    )

    # Optional CSV (ephemeral)
    try:
        import csv
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        csv_path = "/tmp/responses.csv"
        exists = Path(csv_path).exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["timestamp", "first_name", "email", "scores", "top3"])
            w.writerow([
                ts,
                st.session_state.get("first_name", ""),
                st.session_state.get("email", ""),
                json.dumps(scores),
                json.dumps(top3),
            ])
        st.caption("Saved to /tmp/responses.csv (Cloud-safe, ephemeral).")
    except Exception as e:
        st.caption(f"Could not save responses (demo only). {e}")

# Debug tool
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
