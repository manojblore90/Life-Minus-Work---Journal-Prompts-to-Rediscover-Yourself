import os
import re
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

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="centered")
st.title(APP_TITLE)
st.write("Answer 15 questions, add your own reflections, and instantly download a personalized PDF summary.")

# --- Secrets helper: prefer st.secrets, fall back to os.environ ---
def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)

# Pull key + config (prefer secrets)
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # SDK reads from env

HIGH_MODEL = get_secret("OPENAI_HIGH_MODEL", "gpt-5-mini")

def _to_int(s: str, fallback: int) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return fallback

MAX_TOK_HIGH = _to_int(get_secret("MAX_OUTPUT_TOKENS_HIGH", "7000"), 7000)
FALLBACK_CAP = _to_int(get_secret("MAX_OUTPUT_TOKENS_FALLBACK", "6000"), 6000)

USE_AI = bool(OPENAI_API_KEY)

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
                    img = Image.open(p).convert("RGB")  # flatten to RGB (no alpha)
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
    s = (s.replace("’", "'")
         .replace("‘", "'")
         .replace("“", '"')
         .replace("”", '"')
         .replace("–", "-")
         .replace("—", "-")
         .replace("…", "...")
         .replace("•", "*"))
    s = unicodedata.normalize("NFKD", s).encode("latin-1", "ignore").decode("latin-1")
    return s

# ---------- AI call compatibility wrapper ----------
def _call_openai_json(model: str, system: str, user: str, max_tokens: int, temperature: float = 0.7):
    """
    Tries multiple paths so we're compatible with older/newer openai SDKs:
      1) Responses API with response_format
      2) Responses API without response_format
      3) Chat Completions with response_format
      4) Chat Completions without response_format

    Returns: (raw_text, usage_or_None, path_label)
    """
    from openai import OpenAI
    client = OpenAI()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # 1) Responses + response_format
    try:
        r = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return (r.output_text, getattr(r, "usage", None), "responses+rf")
    except TypeError as te:
        if "response_format" not in str(te):
            raise
    except Exception:
        pass

    # 2) Responses without response_format
    try:
        r = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return (r.output_text, getattr(r, "usage", None), "responses")
    except Exception:
        pass

    # 3) Chat Completions + response_format
    try:
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = r.choices[0].message.content if r.choices else ""
        return (content, getattr(r, "usage", None), "chat+rf")
    except TypeError as te:
        if "response_format" not in str(te):
            raise
    except Exception:
        pass

    # 4) Chat Completions without response_format
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = r.choices[0].message.content if r.choices else ""
    return (content, getattr(r, "usage", None), "chat")

# ---------- Optional AI ----------
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
        score_lines = ", ".join([f"{k}: {v}" for k, v in scores.items()])
        packed = [
            {"id": fr["id"], "q": fr["question"], "a": str(fr.get("answer", ""))[:280]}
            for fr in free_responses if fr.get("answer")
        ]
        total_free_chars = sum(len(p.get("a","")) for p in packed)
        model, max_tokens = pick_model(total_free_chars, depth_mode)

        prompt = (
            "You are a warm, practical life coach. Return ONLY valid JSON with keys:\n"
            "  archetype (string), core_need (string),\n"
            "  deep_insight (string, 400-600 words, address the user by first name),\n"
            "  why_now (string, 120-180 words),\n"
            "  strengths (array of 4-6 short strings),\n"
            "  energizers (array of 4), drainers (array of 4),\n"
            "  tensions (array of 2-3 short strings), blindspot (string <= 60 words),\n"
            "  actions (array of EXACTLY 3 short bullet strings),\n"
            "  if_then (array of EXACTLY 3 implementation-intention strings like: 'If it’s 7pm, then I…'),\n"
            "  weekly_plan (array of 7 brief day-plan strings),\n"
            "  affirmation (string <= 15 words), quote (string <= 20 words),\n"
            "  signature_metaphor (string <= 12 words), signature_sentence (string <= 20 words),\n"
            "  top_theme_boosters (array of up to 4 short suggestions), pitfalls (array of up to 4),\n"
            "  future_snapshot (string, 150-220 words, second-person, present tense, written AS IF it is {h} "
            "weeks later and the person followed through),\n"
            "  from_words (object) with: themes (array of EXACTLY 3 short bullets),\n"
            "              quotes (array of 2–3 short verbatim quotes from the user's text, <=12 words each),\n"
            "              insight (string, 80–120 words tying their quotes to top themes),\n"
            "              ritual (one-liner daily ritual drawn from their words),\n"
            "              relationship_moment (one-liner if partner/family appears),\n"
            "              stress_reset (one-liner using their stated reset method),\n"
            "  micro_pledge (string, first-person <= 28 words, derived from their phrases),\n"
            "  weights (object mapping question_id -> object of theme:int in [-2,2]).\n"
            f"User first name: {first_name or 'Friend'}.\n"
            f"Theme scores so far: {score_lines}.\n"
            f"Top 3 themes: {', '.join(top3)}.\n"
            f"Horizon weeks: {horizon_weeks}.\n"
            "Also consider these free-text answers (omit weights for questions you don't see):\n"
            f"{json.dumps(packed, ensure_ascii=False)}\n"
            "Tone: empathetic, encouraging, plain language. No medical/clinical claims. JSON only."
        ).format(h=horizon_weeks)

        system = "Reply with helpful coaching guidance as STRICT JSON only."

        # Primary attempt; if that fails, fallback once with lower cap
        try:
            raw, usage, path = _call_openai_json(model, system, prompt, max_tokens, temperature=0.7)
        except Exception:
            raw, usage, path = _call_openai_json(model, system, prompt, FALLBACK_CAP, temperature=0.7)

        # Show which path was used + usage if available
        try:
            if usage:
                st.caption(
                    f"AI ({path}) • model={model} • input={getattr(usage,'input_tokens','?')} "
                    f"output={getattr(usage,'output_tokens','?')} total={getattr(usage,'total_tokens','?')}"
                )
            else:
                st.caption(f"AI ({path}) • model={model} • max_output_tokens={max_tokens}")
        except Exception:
            pass

        # Clean accidental code fences around JSON
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
            raw = re.sub(r"\n```$", "", raw)

        # Parse JSON (with last-ditch substring extraction)
        try:
            data = json.loads(raw)
        except Exception:
            if "{" in raw and "}" in raw:
                raw2 = raw[raw.find("{"): raw.rfind("}") + 1]
                data = json.loads(raw2)
            else:
                raise

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
            "from_words": {},
            "micro_pledge": str(data.get("micro_pledge", "")),
        }

        fw = data.get("from_words") or {}
        out["from_words"] = {
            "themes": [str(x) for x in (fw.get("themes") or [])][:3],
            "quotes": [str(x) for x in (fw.get("quotes") or [])][:3],
            "insight": str(fw.get("insight", "")),
            "ritual": str(fw.get("ritual", "")),
            "relationship_moment": str(fw.get("relationship_moment", "")),
            "stress_reset": str(fw.get("stress_reset", "")),
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

def left_bar_callout(pdf: FPDF, title: str, body: str, bullets=None):
    if bullets is None:
        bullets = []
    x = pdf.get_x()
    y = pdf.get_y()
    pdf.set_fill_color(30, 144, 255)   # left bar
    pdf.rect(x, y, 2, 6, "F")
    pdf.set_x(x + 4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 6, safe_text(title), ln=True)
    pdf.set_x(x + 4)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, safe_text(body))
    for b in bullets:
        pdf.set_x(x + 4)
        pdf.cell(4, 6, "•")
        pdf.multi_cell(0, 6, safe_text(b))
    pdf.ln(1)

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

    # From your words (new)
    fw = sections.get("from_words") or {}
    if fw and (fw.get("insight") or fw.get("themes") or fw.get("quotes")):
        quotes = [f'"{q}"' for q in fw.get("quotes", []) if q]
        left_bar_callout(pdf, "From your words", fw.get("insight",""), bullets=quotes)
        keep = [("Daily ritual", fw.get("ritual","")),
                ("Connection moment", fw.get("relationship_moment","")),
                ("Stress reset", fw.get("stress_reset",""))]
        if any(v for _, v in keep):
            pdf.set_font("Arial","B",12); pdf.cell(0,6, safe_text("One-liners to keep"), ln=True)
            pdf.set_font("Arial","",12)
            for lbl, val in keep:
                if val: pdf.multi_cell(0,6, safe_text(f"{lbl}: {val}"))
            pdf.ln(1)
    if sections.get("micro_pledge"):
        label_value(pdf, "Personal pledge", sections["micro_pledge"]); pdf.ln(1)

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
# AI status (debug)
with st.expander("AI status (debug)", expanded=False):
    st.write("AI enabled:", USE_AI)
    st.write("Model:", HIGH_MODEL)
    st.write("Max tokens:", MAX_TOK_HIGH, "(fallback", FALLBACK_CAP, ")")
    if not USE_AI:
        st.warning("No OPENAI_API_KEY found. Add it in Settings → Secrets.")
    if USE_AI and st.button("Test OpenAI now"):
        try:
            # Tiny probe to confirm connectivity & JSON roundtrip
            raw, _, path = _call_openai_json(
                HIGH_MODEL,
                "Return strict JSON only.",
                'Return {"ok": true} as JSON.',
                max_tokens=64,
                temperature=0.0,
            )
            st.success(f"OK — via {path}. Output: {raw}")
        except Exception as e:
            st.error(f"OpenAI error: {e}")

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

    questions, _ = load_questions("quest
