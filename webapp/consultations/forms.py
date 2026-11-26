from django import forms

from .models import Consultation, Doctor, disease_choices


def _style(fields, css="form-control"):
    for field in fields.values():
        widget = field.widget
        if isinstance(widget, (forms.CheckboxInput, forms.FileInput)):
            widget.attrs.setdefault("class", "form-check-input" if isinstance(widget, forms.CheckboxInput) else "form-control")
        elif isinstance(widget, forms.Select):
            widget.attrs.setdefault("class", "form-select")
        else:
            widget.attrs.setdefault("class", css)


class DoctorForm(forms.ModelForm):
    """Used by staff in the admin portal to add / edit a doctor profile."""

    class Meta:
        model = Doctor
        fields = [
            "full_name", "specialization", "qualification", "experience_years",
            "clinic_name", "city", "address",
            "consultation_mode", "consultation_fee", "languages",
            "available_days", "available_from", "available_to",
            "phone", "email", "photo", "bio", "is_active",
        ]
        widgets = {
            "address": forms.Textarea(attrs={"rows": 2}),
            "bio": forms.Textarea(attrs={"rows": 3}),
            "available_from": forms.TimeInput(attrs={"type": "time"}),
            "available_to": forms.TimeInput(attrs={"type": "time"}),
            "full_name": forms.TextInput(attrs={"placeholder": "Jane Mathew"}),
            "clinic_name": forms.TextInput(attrs={"placeholder": "Paws & Claws Veterinary Clinic"}),
            "city": forms.TextInput(attrs={"placeholder": "Bengaluru"}),
            "languages": forms.TextInput(attrs={"placeholder": "English, Hindi, Kannada"}),
            "available_days": forms.TextInput(attrs={"placeholder": "Mon - Sat"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _style(self.fields)

    def clean(self):
        cleaned = super().clean()
        start, end = cleaned.get("available_from"), cleaned.get("available_to")
        if start and end and start >= end:
            self.add_error("available_to", "Closing time must be after the opening time.")
        return cleaned


class ConsultationBookingForm(forms.ModelForm):
    """User-facing booking form; disease fields default to the predicted label."""

    predicted_disease = forms.ChoiceField(
        choices=[], required=False, label="Condition detected by AI screening",
    )

    class Meta:
        model = Consultation
        fields = [
            "owner_name", "contact_phone", "contact_email",
            "pet_name", "pet_breed", "pet_age",
            "predicted_disease", "symptoms_noted",
            "consultation_mode", "preferred_date", "preferred_time", "notes",
        ]
        widgets = {
            "symptoms_noted": forms.Textarea(attrs={"rows": 3, "placeholder": "Itching, hair loss, redness..."}),
            "notes": forms.Textarea(attrs={"rows": 3, "placeholder": "Anything else the doctor should know"}),
            "preferred_date": forms.DateInput(attrs={"type": "date"}),
            "preferred_time": forms.TimeInput(attrs={"type": "time"}),
            "pet_age": forms.TextInput(attrs={"placeholder": "2 years 3 months"}),
            "contact_phone": forms.TextInput(attrs={"placeholder": "+91 98765 43210"}),
        }

    def __init__(self, *args, doctor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.doctor = doctor
        self.fields["predicted_disease"].choices = [("", "Not screened / not sure")] + disease_choices()
        if doctor is not None:
            if doctor.consultation_mode != Doctor.CONSULT_BOTH:
                self.fields["consultation_mode"].choices = [
                    c for c in Doctor.CONSULT_MODES if c[0] == doctor.consultation_mode
                ]
                self.initial.setdefault("consultation_mode", doctor.consultation_mode)
            else:
                self.fields["consultation_mode"].choices = [
                    c for c in Doctor.CONSULT_MODES if c[0] != Doctor.CONSULT_BOTH
                ]
        self.fields["contact_email"].required = False
        _style(self.fields)


class ConsultationStatusForm(forms.ModelForm):
    """Staff action on a booking from the admin portal."""

    class Meta:
        model = Consultation
        fields = ["status", "admin_response"]
        widgets = {
            "admin_response": forms.Textarea(
                attrs={"rows": 3, "placeholder": "Message the pet owner will see on their bookings page"}
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _style(self.fields)
