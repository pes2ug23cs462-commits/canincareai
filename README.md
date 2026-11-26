---
title: CanineCare AI
emoji: 🐕
colorFrom: green
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# CanineCare AI — AI-Based Dog Health Monitoring System

Dog skin-disease screening from a photo, using a fine-tuned MobileNetV2 (93.76% accuracy,
2.39M parameters) served through a Django web platform with confidence / Top-3 predictions,
care guidance, a Gemini-powered assistant and nearby-clinic lookup.

PES University — Capstone Project Phase-3 (UE23CS441A).

## Structure
- `webapp/`     — Django application (accounts, dashboard, prediction, chatbot, clinics, consultations)
- `scripts/`    — training, evaluation, benchmarking and diagram scripts
- `notebooks/`  — dataset analysis, model training, results analysis
- `results/`    — model comparison and training artifacts

## Run locally
```bash
cd webapp
pip install -r requirements.txt
# create a .env file with GEMINI_API_KEY and GEOAPIFY_API_KEY
python manage.py migrate
python manage.py runserver
```

> API keys are read from a local `.env` file and are never committed.
