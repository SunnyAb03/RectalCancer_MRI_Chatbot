import json
import os
from typing import Any

from google import genai
from google.genai import types
from openai import OpenAI
try:
    from rag.retriever import get_retriever
except ImportError:
    get_retriever = None

try:
    from zhipuai import ZhipuAI
except ImportError:  # pragma: no cover - optional dependency
    ZhipuAI = None


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDIMuvZMT6jwgHwTKv-jm-1SOoyjwbZkJk")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
GLM_API_KEY = os.getenv("GLM_API_KEY", "2ddfa2ac444c497092c2126ddd3d18c8.dppv4Ruv4ChnCDp8")

# GOOGLE_API_KEY = "AIzaSyDIMuvZMT6jwgHwTKv-jm-1SOoyjwbZkJk"
# GLM_API_KEY = "2ddfa2ac444c497092c2126ddd3d18c8.dppv4Ruv4ChnCDp8"
# OPENAI_API_KEY = "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"


def _ensure_api_key(name: str, value: str) -> None:
    if not value:
        raise RuntimeError(f"Missing required API key: {name}")


def run_extractor_agent(pdf_path: str) -> dict[str, Any]:
    """Agent 1: parse MRI PDF and return strict structured JSON."""
    _ensure_api_key("GOOGLE_API_KEY", GOOGLE_API_KEY)
    client = genai.Client(api_key=GOOGLE_API_KEY)
    uploaded_pdf = client.files.upload(file=pdf_path)

    extract_prompt = """
You are an expert oncological radiologist. Read the attached scanned rectal cancer MRI report.
Return ONLY valid JSON using this exact schema:
{
  "report_summary": "Provide a brief 1-2 sentence clinical summary of the overall MRI findings.",
  "tumor_location": "Describe location",
  "t_stage": "T stage or 'Not specified'",
  "n_stage": "N stage or 'Not specified'",
  "crm_status": "'Threatened', 'Involved', 'Clear', or 'Not mentioned'",
  "emvi_status": "'Positive', 'Negative', or 'Not mentioned'",
  "mrtrg_score": "mrTRG score, else 'Not applicable'",
  "tumor_deposits": "'Present', 'Absent', or 'Not mentioned'"
}
Do not include markdown, prose, or extra keys.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_pdf, extract_prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )

    try:
        parsed = json.loads(response.text or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Extractor returned invalid JSON.") from exc

    required_keys = {
        "report_summary",
        "tumor_location",
        "t_stage",
        "n_stage",
        "crm_status",
        "emvi_status",
        "mrtrg_score",
        "tumor_deposits",
    }
    missing = required_keys - set(parsed.keys())
    if missing:
        raise RuntimeError(f"Extractor JSON missing required keys: {sorted(missing)}")
    return parsed


def run_summarizer_agent(extracted_json: dict[str, Any], language: str) -> str:
    """Agent 2: produce a short patient-friendly summary."""
    _ensure_api_key("GOOGLE_API_KEY", GOOGLE_API_KEY)
    client = genai.Client(api_key=GOOGLE_API_KEY)

    prompt = f"""
You are a clinical nurse specialist.
Write exactly 1-2 short sentences in {language}.
Use UK NHS 9-11 year old reading level.
Explain this MRI result in simple words.
Never predict prognosis, survival, or recurrence.
MRI JSON:
{json.dumps(extracted_json, ensure_ascii=False)}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return (response.text or "").strip()


def _chat_system_prompt(language: str = "English", literature_context: str = "") -> str:
    base = f"""
You are an empathetic, supportive AI assistant designed to help patients understand their rectal cancer MRI report.

CRITICAL INSTRUCTION: You MUST communicate with the patient in {language}.

STRICT SAFETY RULES:
1. Role Clarity: You are an AI Tool, NOT a doctor or a Clinical Nurse Specialist. If a patient asks for medical advice or diagnosis, gently remind them of your role as an AI assistant.
2. Target Reading Level: 9-11 years old (UK NHS standard). Use short, simple sentences and everyday analogies.
3. NO PROGNOSTIC PREDICTIONS: NEVER predict survival rates, prognosis, life expectancy, or cancer recurrence.
4. Do not give direct treatment recommendations, but you can explain what standard terms mean.
5. If the user asks something completely outside the provided data or clinical glossary, gently reply: "I cannot tell from the report you provided, please discuss this with your doctor."
6. FORMATTING: You MUST use Markdown to structure your response. Break your answers into short paragraphs (no more than 3-4 sentences each). Use bullet points when listing items, and use **bold text** to highlight key medical terms. NEVER output a single massive block of text.

CLINICAL GLOSSARY & EXPLANATION GUIDELINES (Based on Cancer Research UK):

- T-Stage (Tumour): Explain this describes "how far the tumour has grown through the wall of your bowel."
- Tis: The earliest stage, only in the bowel lining.
- T1: The tumour has grown into the inner layer of the bowel wall, known as the submucosa.
- T2: The tumour has grown past the submucosa and is now invading a muscular layer of the bowel wall called muscularis propria.
- T3: The tumour has grown through the muscle layer and into the surrounding tissues/fat. T3 tumours are broken down further (T3a: <1 mm, T3b: 1-5 mm, T3c: 5-15 mm, T3d: >15 mm beyond the muscle layer).
- T4a: It has grown completely through the bowel wall and is not touching nearby organs.
- T4b: It has grown completely through the bowel wall and is touching nearby organs.

- N-Stage (Node): Explain these are "lymph nodes, which are like tiny filters in our body that protect us from infections."
- N0: No cancer cells in the lymph nodes.
- N1a / N1b: Cancer cells found in 1 to 3 nearby lymph nodes.
- N2: Cancer cells found in 4 or more nearby lymph nodes.

- Tumour Deposit (N1c): CRITICAL DISTINCTION - N1c does NOT refer to lymph nodes. Explain these are "small clusters or dots of cancer cells found in the fat around the bowel." These deposits are often linked to EMVI and indicate cancer cells that have begun the process of spreading.

- M-Stage (Metastasis): Explain this describes "whether the cancer has spread to a different part of the body (like the liver or lungs)." Remember safety rule #3: do not predict prognosis.

- CRM (Circumferential Resection Margin): Explain this is the "safety margin" or border around the bowel. Doctors look at this to plan how to safely remove the tumour.

- EMVI (Extramural Venous Invasion): Explain this as "cancer cells being spotted inside the tiny blood vessels just outside the bowel wall."

- TME (Total Mesorectal Excision): The "standard, highly precise surgical technique used to carefully remove the rectum and the package of fat surrounding it."

- 'y' prefix (e.g., yT, yN): Explain that "the 'y' simply means this scan was done *after* having some treatment like chemotherapy or radiotherapy."

- MDT (Multidisciplinary Team): Explain that "Every single patient is discussed by an MDT. This is a large team of different experts who meet together to agree on the absolute best, personalised treatment plan for you."

ADDITIONAL KNOWLEDGE & EMOTIONAL SUPPORT (Based on Macmillan & Bowel Cancer UK):
- The Rectum/Bowel: If asked where the cancer is, explain that "the bowel is part of your digestive system, and the rectum is the very last part of the large bowel, just before your bottom."
- Polyps: If polyps are mentioned, explain they are "small growths on the inner lining of the bowel. They are often non-cancerous, but doctors check them carefully."
- Emotional Support & Signposting: If the patient expresses fear, extreme anxiety, or says they are overwhelmed, you MUST first validate their feelings. Then, gently remind them of their MDT, and suggest they can reach out to charities like Macmillan Cancer Support and Bowel Cancer UK.
- CREATIVE ANALOGIES (CRITICAL): The clinical glossary above is ONLY for your factual reference. DO NOT just copy and paste it. You MUST explain these concepts using your own native conversational style.
"""

    if literature_context:
        base += f"""

RELEVANT CLINICAL LITERATURE (FOR BACKGROUND REFERENCE ONLY):
---
{literature_context}
---

CRITICAL LITERATURE USAGE RULES — VIOLATING THESE IS UNSAFE:
L-1. The literature above describes GENERAL POPULATION statistics from published studies. It does NOT describe this specific patient.
L-2. You MAY use these excerpts to explain what a medical term means, or to describe what is generally known about a condition (e.g., "studies have shown that CRM status is an important factor in surgical planning").
L-3. You MUST NEVER use these statistics to make predictions about this individual patient's prognosis, survival rate, recurrence risk, or treatment outcome.
L-4. You MUST NEVER say phrases like "based on the literature, your prognosis is..." or "studies suggest your outcome will be..." or "your survival rate based on these studies is..."
L-5. If the patient asks a question that would require personal prognosis, you MUST respond: "I cannot predict individual outcomes using research statistics. Every patient is unique. Please discuss your personal outlook with your MDT."
L-6. When referencing literature, you MUST attribute the source (e.g., "according to one published study...") and preface with "In general..." or "Across populations..." to make clear you are NOT describing their case.
L-7. Do NOT cite specific numerical statistics from the literature to the patient (percentages, hazard ratios, p-values). Translate into plain language qualifiers like "common", "less common", "important for doctors to monitor".
"""
    return base


def run_chat_agent(
    user_message: str,
    extracted_json: dict[str, Any],
    chat_history: list[dict[str, str]],
    model_choice: str,
) -> dict[str, Any]:
    """Agent 3: route patient Q&A to selected model, enriched with RAG literature context."""

    # --- Retrieve relevant literature context ---
    literature_context = ""
    if get_retriever is not None:
        try:
            retriever = get_retriever()
            literature_context = retriever.format_context(user_message)
        except Exception:
            literature_context = ""  # graceful degradation — RAG is optional

    system_prompt = _chat_system_prompt(language="English", literature_context=literature_context)

    context = (
        "MRI extracted JSON:\n"
        f"{json.dumps(extracted_json, ensure_ascii=False)}\n\n"
        f"Patient question:\n{user_message}"
    )

    answer: str
    if model_choice == "Gemini":
        _ensure_api_key("GOOGLE_API_KEY", GOOGLE_API_KEY)
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_prompt, context],
            config=types.GenerateContentConfig(temperature=0.3),
        )
        answer = (response.text or "").strip()

    elif model_choice == "OpenAI":
        _ensure_api_key("OPENAI_API_KEY", OPENAI_API_KEY)
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            temperature=0.3,
        )
        answer = (response.choices[0].message.content or "").strip()

    elif model_choice == "ZhipuAI":
        if ZhipuAI is None:
            raise RuntimeError("zhipuai package is not installed.")
        _ensure_api_key("GLM_API_KEY", GLM_API_KEY)
        client = ZhipuAI(api_key=GLM_API_KEY)
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            temperature=0.3,
        )
        answer = (response.choices[0].message.content or "").strip()
    else:
        raise ValueError("model_choice must be one of: Gemini, ZhipuAI, OpenAI")

    return {
        "answer": answer,
        "metadata": {},
    }
