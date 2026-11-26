from django.contrib import admin

from .models import Consultation, Doctor


@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    list_display = ("full_name", "specialization", "city", "experience_years", "consultation_fee", "is_active")
    list_filter = ("is_active", "city", "consultation_mode")
    search_fields = ("full_name", "city", "specialization", "clinic_name")


@admin.register(Consultation)
class ConsultationAdmin(admin.ModelAdmin):
    list_display = ("pet_name", "owner_name", "doctor", "predicted_display_name", "preferred_date", "status", "is_seen_by_admin")
    list_filter = ("status", "is_seen_by_admin", "predicted_disease")
    search_fields = ("pet_name", "owner_name", "doctor__full_name")
    readonly_fields = ("created_at", "updated_at")
