"""Doctor directory and consultation bookings.

Doctors are created and maintained by staff through the admin portal
(/manage/). Registered users browse that directory, filter it by experience
and place, and book a consultation - the booking form is pre-filled with the
label and disease information produced by the last screening the user ran.
"""
from django.contrib.auth.models import User
from django.db import models
from django.urls import reverse

from dog_disease_project.data_loader import get_class_names, get_disease_info


def disease_choices():
    """Model class labels, shown with their friendly display names."""
    info = get_disease_info()
    return [
        (name, info.get(name, {}).get("display_name", name.replace("_", " ").title()))
        for name in get_class_names()
    ]


class Doctor(models.Model):
    """A veterinary doctor a user can book a consultation with."""

    CONSULT_ONLINE = "online"
    CONSULT_CLINIC = "clinic"
    CONSULT_BOTH = "both"
    CONSULT_MODES = [
        (CONSULT_CLINIC, "In-clinic visit"),
        (CONSULT_ONLINE, "Online consultation"),
        (CONSULT_BOTH, "In-clinic & online"),
    ]

    full_name = models.CharField(max_length=150)
    specialization = models.CharField(
        max_length=120,
        default="Veterinary Dermatology",
        help_text="e.g. Veterinary Dermatology, Small Animal Medicine",
    )
    qualification = models.CharField(max_length=150, blank=True, help_text="e.g. BVSc & AH, MVSc")
    experience_years = models.PositiveIntegerField(default=0, help_text="Total years of practice")

    clinic_name = models.CharField(max_length=150, blank=True)
    city = models.CharField(max_length=100, help_text="City / place the user can consult at")
    address = models.TextField(blank=True)

    consultation_mode = models.CharField(max_length=10, choices=CONSULT_MODES, default=CONSULT_BOTH)
    consultation_fee = models.DecimalField(max_digits=8, decimal_places=2, default=0)
    languages = models.CharField(max_length=150, blank=True, help_text="Comma separated, e.g. English, Hindi")
    available_days = models.CharField(max_length=150, blank=True, help_text="e.g. Mon-Fri")
    available_from = models.TimeField(null=True, blank=True)
    available_to = models.TimeField(null=True, blank=True)

    phone = models.CharField(max_length=20, blank=True)
    email = models.EmailField(blank=True)
    photo = models.ImageField(upload_to="doctors/", blank=True, null=True)
    bio = models.TextField(blank=True)

    is_active = models.BooleanField(default=True, help_text="Only active doctors are listed to users")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-experience_years", "full_name"]

    def __str__(self):
        return f"Dr. {self.full_name} ({self.city})"

    def get_absolute_url(self):
        return reverse("consultations:doctor_detail", args=[self.pk])

    @property
    def display_name(self):
        name = self.full_name.strip()
        return name if name.lower().startswith("dr") else f"Dr. {name}"

    @property
    def initials(self):
        parts = [p for p in self.full_name.replace("Dr.", "").split() if p]
        return "".join(p[0].upper() for p in parts[:2]) or "V"

    @property
    def language_list(self):
        return [lang.strip() for lang in self.languages.split(",") if lang.strip()]

    @property
    def experience_band(self):
        """Bucket used by the experience filter on the user-facing list."""
        if self.experience_years >= 15:
            return "15+"
        if self.experience_years >= 10:
            return "10+"
        if self.experience_years >= 5:
            return "5+"
        return "0+"

    @property
    def availability_text(self):
        if self.available_from and self.available_to:
            window = f"{self.available_from.strftime('%I:%M %p')} - {self.available_to.strftime('%I:%M %p')}"
            return f"{self.available_days or 'Daily'}, {window}"
        return self.available_days or "Contact clinic for timings"


class Consultation(models.Model):
    """A booking a user raised against a doctor, carrying the AI screening result."""

    STATUS_PENDING = "pending"
    STATUS_CONFIRMED = "confirmed"
    STATUS_COMPLETED = "completed"
    STATUS_CANCELLED = "cancelled"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_CONFIRMED, "Confirmed"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_CANCELLED, "Cancelled"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="consultations")
    doctor = models.ForeignKey(Doctor, on_delete=models.PROTECT, related_name="consultations")

    owner_name = models.CharField(max_length=150)
    contact_phone = models.CharField(max_length=20)
    contact_email = models.EmailField(blank=True)

    pet_name = models.CharField(max_length=100)
    pet_breed = models.CharField(max_length=100, blank=True)
    pet_age = models.CharField(max_length=50, blank=True, help_text="e.g. 2 years 3 months")

    # Pre-filled from the user's most recent screening result.
    predicted_disease = models.CharField(max_length=100, blank=True)
    predicted_display_name = models.CharField(max_length=150, blank=True)
    predicted_confidence = models.FloatField(null=True, blank=True)
    predicted_severity = models.CharField(max_length=100, blank=True)
    disease_summary = models.TextField(blank=True, help_text="Disease description carried over from the screening")
    symptoms_noted = models.TextField(blank=True)

    consultation_mode = models.CharField(max_length=10, choices=Doctor.CONSULT_MODES, default=Doctor.CONSULT_CLINIC)
    preferred_date = models.DateField()
    preferred_time = models.TimeField(null=True, blank=True)
    notes = models.TextField(blank=True)

    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default=STATUS_PENDING)
    admin_response = models.TextField(blank=True, help_text="Reply shown to the user on their bookings page")
    is_seen_by_admin = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.pet_name} with {self.doctor.display_name} on {self.preferred_date}"

    @property
    def status_css(self):
        return {
            self.STATUS_PENDING: "warning",
            self.STATUS_CONFIRMED: "primary",
            self.STATUS_COMPLETED: "success",
            self.STATUS_CANCELLED: "danger",
        }.get(self.status, "secondary")

    @property
    def is_new(self):
        """Drives the highlighted notification in the admin portal."""
        return not self.is_seen_by_admin
