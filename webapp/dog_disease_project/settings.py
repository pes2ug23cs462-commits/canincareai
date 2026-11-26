"""
Django settings for the CanineCare AI project.
"""

import os
from pathlib import Path

import dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

dotenv.load_dotenv(BASE_DIR / ".env")


def env_bool(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


SECRET_KEY = os.environ.get("SECRET_KEY", "django-insecure-fallback-key-for-local-dev-only")

DEBUG = env_bool("DEBUG", True)

ALLOWED_HOSTS = [
    h.strip() for h in os.environ.get("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if h.strip()
]

# Project-level configuration read from .env (never hard-code secrets here).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEOAPIFY_API_KEY = os.environ.get("GEOAPIFY_API_KEY", "")

# Paths to the Stage 1 artifacts used by the app.
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

PROPOSED_MODEL_PATH = MODELS_DIR / "dog_disease_proposed_model.pt"
CLASS_NAMES_PATH = DATA_DIR / "class_names.json"
DISEASE_INFO_PATH = DATA_DIR / "disease_information.json"
DATASET_STATS_PATH = DATA_DIR / "dataset_stats.json"
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
TRAINING_HISTORY_JSON = RESULTS_DIR / "training_history.json"
PREDICTIONS_JSON = RESULTS_DIR / "predictions.json"
LATENCY_BENCHMARK_JSON = RESULTS_DIR / "latency_benchmark.json"

# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "accounts",
    "dashboard",
    "prediction",
    "chatbot",
    "clinics",
    "consultations",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "dog_disease_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "consultations.context_processors.admin_notifications",
            ],
        },
    },
]

WSGI_APPLICATION = "dog_disease_project.wsgi.application"


# Database

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# Password validation

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# Internationalization

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True


# Static & media files

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Email (console backend for local/demo use - no real emails are sent)
EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"

# Auth redirects
LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "dashboard:home"
LOGOUT_REDIRECT_URL = "accounts:login"

MAX_UPLOAD_SIZE_MB = 10
