"""Staff-only admin portal routes, mounted at /manage/ under the "manage" namespace."""
from django.urls import path

from . import views

app_name = "manage"

urlpatterns = [
    path("", views.admin_dashboard_view, name="dashboard"),
    path("doctors/", views.admin_doctor_list_view, name="doctors"),
    path("doctors/add/", views.admin_doctor_create_view, name="doctor_add"),
    path("doctors/<int:pk>/edit/", views.admin_doctor_edit_view, name="doctor_edit"),
    path("doctors/<int:pk>/toggle/", views.admin_doctor_toggle_view, name="doctor_toggle"),
    path("doctors/<int:pk>/delete/", views.admin_doctor_delete_view, name="doctor_delete"),
    path("consultations/", views.admin_consultation_list_view, name="consultations"),
    path("consultations/<int:pk>/", views.admin_consultation_detail_view, name="consultation_detail"),
    path("notifications/clear/", views.admin_mark_all_seen_view, name="mark_all_seen"),
]
