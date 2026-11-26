"""URL configuration for the CanineCare AI project."""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("accounts.urls")),
    path("chatbot/", include("chatbot.urls")),
    path("clinics/", include("clinics.urls")),
    path("consult/", include("consultations.urls")),
    path("manage/", include("consultations.admin_urls")),
    path("predict/", include("prediction.urls")),
    path("", include("dashboard.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.BASE_DIR / "static")
