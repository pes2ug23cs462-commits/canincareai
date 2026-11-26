FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DJANGO_SETTINGS_MODULE=dog_disease_project.settings

WORKDIR /app

COPY webapp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp/ /app/

RUN python manage.py collectstatic --noinput || true

EXPOSE 7860
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn dog_disease_project.wsgi:application --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 180"]
