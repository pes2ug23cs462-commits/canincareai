"""Domain-restricted Gemini chatbot service.

The Gemini API key is read only from the environment (.env), never hard-coded.
"""
import json

import requests
from django.conf import settings

from dog_disease_project.data_loader import get_disease_info

GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

SYSTEM_PROMPT = """You are "CanineCare AI Assistant", a domain-restricted assistant embedded in a \
dog skin disease detection application.

STRICT DOMAIN RESTRICTION:
You may ONLY answer questions related to: dog diseases, pet diseases, symptoms, prevention, \
general veterinary care, the disease information in this application's knowledge base, \
interpreting this application's AI predictions, and when veterinary attention may be appropriate.

If a user asks about anything outside this domain (general programming, unrelated trivia, \
politics, entertainment, math homework, etc.), you MUST refuse briefly and redirect, for example: \
"I can only assist with pet and dog health-related questions." Do not answer the unrelated \
question in any way, even partially.

SAFETY RULES (always follow):
- Never provide a definitive diagnosis. You may discuss possibilities and general information only.
- Never prescribe specific drugs or dosages.
- Always encourage consulting a licensed veterinarian for real diagnosis and treatment.
- If the user describes symptoms that could be an emergency (difficulty breathing, facial \
swelling, collapse, uncontrolled bleeding, suspected poisoning, seizures), tell them to seek \
immediate veterinary attention rather than trying to diagnose the issue yourself.
- You are not a veterinarian and must never claim to be one.
- Clearly distinguish this application's AI predictions (a screening tool, not a diagnosis) \
from an actual veterinary diagnosis.

FORMATTING (strict):
- Keep the entire answer short: 2-5 sentences, or a short list of at most 5 bullet points.
- Do not use markdown headings (### or ##), tables, or numbered outlines with sub-bullets.
- Use plain paragraphs or simple "- " bullet points only, no bold headers before every bullet.
- Be direct - skip lengthy preambles.

Use the disease knowledge base context provided below when relevant.

DISEASE KNOWLEDGE BASE (from this application's dataset):
{knowledge_base}
"""


class ChatbotError(Exception):
    pass


def _build_system_prompt():
    kb = get_disease_info()
    kb_text = json.dumps(kb, indent=2)
    return SYSTEM_PROMPT.format(knowledge_base=kb_text)


def ask_gemini(user_message, history=None):
    """Send a message to Gemini with the domain-restricted system prompt.

    `history` is an optional list of {"role": "user"|"model", "text": str} for
    short conversational context.
    """
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise ChatbotError(
            "The chatbot is not configured yet. Add GEMINI_API_KEY to the .env file to enable it."
        )

    contents = []
    for turn in (history or [])[-6:]:
        contents.append({
            "role": "user" if turn["role"] == "user" else "model",
            "parts": [{"text": turn["text"]}],
        })
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "system_instruction": {"parts": [{"text": _build_system_prompt()}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048},
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            params={"key": api_key},
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise ChatbotError(f"Could not reach the chatbot service: {exc}") from exc

    if resp.status_code != 200:
        raise ChatbotError(f"Chatbot service error ({resp.status_code}). Please try again shortly.")

    data = resp.json()
    try:
        candidate = data["candidates"][0]
        # With extended-thinking models, earlier parts may be internal "thought"
        # content - use the last text part, which holds the actual answer.
        text_parts = [p["text"] for p in candidate["content"]["parts"] if p.get("text") and not p.get("thought")]
        if not text_parts:
            raise ChatbotError("The chatbot did not return a usable response. Please try again.")
        reply = text_parts[-1].strip()
        if candidate.get("finishReason") == "MAX_TOKENS" and not reply:
            raise ChatbotError("The response was too long to complete. Please ask a more specific question.")
        return reply
    except (KeyError, IndexError):
        raise ChatbotError("The chatbot did not return a usable response. Please try again.")
