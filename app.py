import os
import json
import base64
import uuid
from typing import Dict, Any, List
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter

# ── Environment Setup ────────────────────────────────────────
load_dotenv(override=True)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="HackEval AI",
    page_icon="🏆",
    layout="centered",
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── System Logic ──────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_converter = DocumentConverter()

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_document(file_path: str) -> str:
    """Extracts text from PDF or other documents using Docling."""
    try:
        result = _converter.convert(file_path)
        return result.document.export_to_markdown()
    except Exception as e:
        print(f"Error parsing document: {e}")
        return ""

def evaluate_submission(data: Dict[str, str], file_path: str) -> Dict[str, Any]:
    """Evaluates the hackathon submission using GPT-4o (Multi-modal)."""
    is_image = file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    
    submission_context = f"""
    Topic: {data['topic']}
    Problem Statement: {data['problem']}
    Approach: {data['approach']}
    Practical Application: {data['application']}
    Impact & Scalability: {data['impact']}
    """

    messages = [
        {
            "role": "system",
            "content": "You are an expert hackathon judge and AI analyst. Your task is to evaluate submissions for technical viability, realism, and alignment between text and provided visuals."
        }
    ]

    user_content = [
        {
            "type": "text",
            "text": f"Evaluate this hackathon submission based on the following details:\n{submission_context}\n\n"
                    "Specifically look for AI-generated hallmarks: \n"
                    "1. Overly structured or 'perfect' academic tone with no variety in sentence length.\n"
                    "2. Lack of specific local nuances or 'messy' real-world constraints.\n"
                    "3. Standard AI ending patterns (summarizing the impact in a predictable way).\n"
                    "4. Generic, high-level vocabulary that avoids technical depth.\n\n"
                    "Check if the provided file (image or text extracted from document) aligns with these points. "
                    "If it does not align, reject it. Provide a realistic 'ai_generated_probability' (0-100)."
        }
    ]

    if is_image:
        base64_image = encode_image(file_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    else:
        extracted_text = parse_document(file_path)
        user_content[0]["text"] += f"\n\nExtracted Text from Document:\n{extracted_text}"

    messages.append({"role": "user", "content": user_content})

    prompt_extension = """
    Return ONLY a JSON object with the following structure:
    {
      "status": "ACCEPTED" or "REJECTED",
      "reason_for_rejection": "Details if rejected",
      "scores": {
        "practical_viability": 1-10,
        "problem_realism": 1-10,
        "approach_logic": 1-10,
        "impact_scalability": 1-10
      },
      "ai_generated_probability": 0-100,
      "alignment_score": 0-100,
      "summary": "Full evaluation summary"
    }
    """
    user_content[0]["text"] += prompt_extension

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2
    )

    result = json.loads(response.choices[0].message.content)
    
    if result.get("alignment_score", 0) < 50:
        result["status"] = "REJECTED"
        result["reason_for_rejection"] = result.get("reason_for_rejection") or "The uploaded file does not align with the submission details."
        
    if result.get("ai_generated_probability", 0) > 75:
        result["status"] = "REJECTED"
        result["reason_for_rejection"] = f"Submission flagged as likely AI-generated ({result['ai_generated_probability']}% probability)."

    return result

# ── Custom UI CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { max-width: 720px; padding-top: 2rem; }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1.2rem; color: white;
        text-align: center; margin-bottom: 0.5rem;
    }
    .score-card h2 { margin: 0; font-size: 2rem; }
    .score-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .status-accepted {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 0.8rem 1.5rem; border-radius: 10px; text-align:center;
        font-weight:700; font-size:1.1rem; color:#064e3b;
    }
    .status-rejected {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.8rem 1.5rem; border-radius: 10px; text-align:center;
        font-weight:700; font-size:1.1rem; color:#4a0020;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("# 🏆 HackEval AI")
st.caption("AI-Powered Hackathon Submission Evaluator")
st.divider()

# ── Submission Form ──────────────────────────────────────────
with st.form("submission_form"):
    topic = st.text_input("Project Topic", max_chars=100,
                          placeholder="e.g. Smart Water Management System")
    problem = st.text_area("Problem Statement", height=100, max_chars=500,
                           placeholder="What real-world problem does your project solve? (Max 500 chars)")
    approach = st.text_area("Approach / Solution", height=100, max_chars=500,
                            placeholder="Describe how your solution works technically. (Max 500 chars)")
    application = st.text_area("Practical Application", height=80, max_chars=500,
                               placeholder="Where and how can this be applied? (Max 500 chars)")
    impact = st.text_area("Impact & Scalability", height=80, max_chars=500,
                          placeholder="What impact does it create and how will it scale? (Max 500 chars)")
    file = st.file_uploader("Upload Supporting File",
                            type=["pdf", "png", "jpg", "jpeg", "webp"],
                            help="Max file size: 5MB. PDF or Image.")

    submitted = st.form_submit_button("🚀  Evaluate Submission", use_container_width=True)

# ── Evaluation Logic ─────────────────────────────────────────
if submitted:
    if not all([topic, problem, approach, application, impact]) or file is None:
        st.error("Please fill in all fields and upload a supporting file.")
    elif file.size > 5 * 1024 * 1024:
        st.error("❌ File size exceeds 5MB limit.")
    else:
        file_ext = os.path.splitext(file.name)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        data = {"topic": topic, "problem": problem, "approach": approach, "application": application, "impact": impact}

        with st.status("🚀 Evaluating your submission...", expanded=True) as status:
            try:
                status.write("📄 Extracting text from document...")
                result = evaluate_submission(data, file_path)
                status.update(label="✅ Evaluation Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Evaluation Failed", state="error")
                st.error(f"Error: {e}")
                st.stop()
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        st.divider()
        st.subheader("📊 Evaluation Results")

        res_status = result.get("status", "UNKNOWN")
        if res_status == "ACCEPTED":
            st.markdown('<div class="status-accepted">✅ ACCEPTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-rejected">❌ REJECTED</div>', unsafe_allow_html=True)
            if result.get("reason_for_rejection"):
                st.warning(f"**Reason:** {result['reason_for_rejection']}")

        st.write("")
        scores = result.get("scores", {})
        cols = st.columns(4)
        score_labels = [
            ("practical_viability", "Is it Practically Viable?"),
            ("problem_realism", "Is the Problem Real?"),
            ("approach_logic", "Is the Approach Logical?"),
            ("impact_scalability", "Are there practical applications?"),
        ]
        for col, (key, label) in zip(cols, score_labels):
            val = scores.get(key, "–")
            col.markdown(f'<div class="score-card"><h2>{val}</h2><p>{label}</p></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("🚫 AI-Generated Probability", f"{result.get('ai_generated_probability', '–')}%")
        c2.metric("🎯 Alignment Score", f"{result.get('alignment_score', '–')}%")

        if result.get("summary"):
            st.markdown("### 📝 Summary")
            st.info(result["summary"])
