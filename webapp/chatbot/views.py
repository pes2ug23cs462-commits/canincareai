import json

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST

from .gemini_service import ChatbotError, ask_gemini

CHAT_SESSION_KEY = "chatbot_history"


@login_required
def chatbot_view(request):
    return render(request, "chatbot.html", {"configured": bool(settings.GEMINI_API_KEY)})


@login_required
@require_POST
@csrf_protect
def chat_api(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid request."}, status=400)

    message = (payload.get("message") or "").strip()
    if not message:
        return JsonResponse({"error": "Please enter a message."}, status=400)
    if len(message) > 1000:
        return JsonResponse({"error": "Message is too long (max 1000 characters)."}, status=400)

    history = request.session.get(CHAT_SESSION_KEY, [])

    try:
        reply = ask_gemini(message, history=history)
    except ChatbotError as exc:
        return JsonResponse({"error": str(exc)}, status=503)

    history.append({"role": "user", "text": message})
    history.append({"role": "model", "text": reply})
    request.session[CHAT_SESSION_KEY] = history[-12:]

    return JsonResponse({"reply": reply})
