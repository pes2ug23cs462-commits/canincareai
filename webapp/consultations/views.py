"""User-facing doctor directory / booking flow and the staff admin portal."""
from datetime import date

from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from dog_disease_project.data_loader import get_disease_info

from .forms import ConsultationBookingForm, ConsultationStatusForm, DoctorForm
from .models import Consultation, Doctor

EXPERIENCE_FILTERS = [
    ("", "Any experience"),
    ("5", "5+ years"),
    ("10", "10+ years"),
    ("15", "15+ years"),
]

# Non-staff visitors are sent to the login page with the Admin tab pre-selected.
staff_required = user_passes_test(
    lambda u: u.is_active and u.is_staff, login_url="/accounts/login/?role=admin"
)


# --------------------------------------------------------------------------
# User-facing: browse doctors, book a consultation, track bookings
# --------------------------------------------------------------------------

def _session_prediction(request):
    """Last screening result stashed by the prediction view, if any."""
    return request.session.get("last_prediction") or {}


@login_required
def doctor_list_view(request):
    doctors = Doctor.objects.filter(is_active=True)

    query = request.GET.get("q", "").strip()
    city = request.GET.get("city", "").strip()
    experience = request.GET.get("experience", "").strip()
    mode = request.GET.get("mode", "").strip()
    sort = request.GET.get("sort", "experience").strip()

    if query:
        doctors = doctors.filter(
            Q(full_name__icontains=query)
            | Q(specialization__icontains=query)
            | Q(clinic_name__icontains=query)
            | Q(city__icontains=query)
        )
    if city:
        doctors = doctors.filter(city__iexact=city)
    if experience.isdigit():
        doctors = doctors.filter(experience_years__gte=int(experience))
    if mode:
        doctors = doctors.filter(Q(consultation_mode=mode) | Q(consultation_mode=Doctor.CONSULT_BOTH))

    if sort == "fee":
        doctors = doctors.order_by("consultation_fee", "-experience_years")
    elif sort == "name":
        doctors = doctors.order_by("full_name")
    else:
        doctors = doctors.order_by("-experience_years", "full_name")

    cities = (
        Doctor.objects.filter(is_active=True)
        .exclude(city="")
        .values_list("city", flat=True)
        .order_by("city")
        .distinct()
    )

    context = {
        "doctors": doctors,
        "cities": cities,
        "experience_filters": EXPERIENCE_FILTERS,
        "modes": Doctor.CONSULT_MODES,
        "selected": {"q": query, "city": city, "experience": experience, "mode": mode, "sort": sort},
        "prediction": _session_prediction(request),
        "total_active": Doctor.objects.filter(is_active=True).count(),
    }
    return render(request, "consultations/doctor_list.html", context)


@login_required
def doctor_detail_view(request, pk):
    doctor = get_object_or_404(Doctor, pk=pk, is_active=True)
    return render(request, "consultations/doctor_detail.html", {
        "doctor": doctor,
        "prediction": _session_prediction(request),
    })


@login_required
def book_consultation_view(request, pk):
    doctor = get_object_or_404(Doctor, pk=pk, is_active=True)
    prediction = _session_prediction(request)

    profile_name = getattr(getattr(request.user, "profile", None), "full_name", "")
    initial = {
        "owner_name": profile_name or request.user.get_full_name() or request.user.username,
        "contact_email": request.user.email,
        "preferred_date": date.today().isoformat(),
        "consultation_mode": (
            doctor.consultation_mode if doctor.consultation_mode != Doctor.CONSULT_BOTH else Doctor.CONSULT_CLINIC
        ),
    }
    # Default the clinical fields to whatever the AI screening predicted.
    if prediction:
        initial["predicted_disease"] = prediction.get("predicted_class", "")
        initial["symptoms_noted"] = "\n".join(prediction.get("symptoms", [])[:4])

    if request.method == "POST":
        form = ConsultationBookingForm(request.POST, doctor=doctor)
        if form.is_valid():
            consultation = form.save(commit=False)
            consultation.user = request.user
            consultation.doctor = doctor

            # Carry the screening context onto the booking so the doctor and the
            # admin portal see exactly what the model reported.
            label = consultation.predicted_disease
            if label:
                info = get_disease_info().get(label, {})
                consultation.predicted_display_name = info.get("display_name", label.replace("_", " ").title())
                consultation.disease_summary = info.get("description", "")
                consultation.predicted_severity = info.get("severity", "")
                if prediction.get("predicted_class") == label:
                    consultation.predicted_confidence = prediction.get("confidence")
                    consultation.predicted_severity = prediction.get("severity") or consultation.predicted_severity

            consultation.save()
            messages.success(
                request,
                f"Consultation request sent to {doctor.display_name}. "
                "You will see the confirmation on your bookings page once it is reviewed.",
            )
            return redirect("consultations:my_bookings")
        messages.error(request, "Please correct the highlighted fields and submit again.")
    else:
        form = ConsultationBookingForm(initial=initial, doctor=doctor)

    return render(request, "consultations/book_consultation.html", {
        "doctor": doctor,
        "form": form,
        "prediction": prediction,
        "today": date.today().isoformat(),
    })


@login_required
def my_bookings_view(request):
    bookings = Consultation.objects.filter(user=request.user).select_related("doctor")
    return render(request, "consultations/my_bookings.html", {
        "bookings": bookings,
        "upcoming": bookings.filter(
            status__in=[Consultation.STATUS_PENDING, Consultation.STATUS_CONFIRMED],
            preferred_date__gte=date.today(),
        ).count(),
    })


@login_required
def cancel_booking_view(request, pk):
    booking = get_object_or_404(Consultation, pk=pk, user=request.user)
    if request.method == "POST":
        if booking.status in (Consultation.STATUS_COMPLETED, Consultation.STATUS_CANCELLED):
            messages.info(request, "This booking can no longer be cancelled.")
        else:
            booking.status = Consultation.STATUS_CANCELLED
            booking.is_seen_by_admin = False  # re-notify the portal about the change
            booking.save(update_fields=["status", "is_seen_by_admin", "updated_at"])
            messages.success(request, "Your consultation has been cancelled.")
    return redirect("consultations:my_bookings")


# --------------------------------------------------------------------------
# Admin portal (staff only)
# --------------------------------------------------------------------------

@staff_required
def admin_dashboard_view(request):
    consultations = Consultation.objects.select_related("doctor", "user")
    new_requests = consultations.filter(is_seen_by_admin=False)

    counts = {row["status"]: row["n"] for row in consultations.values("status").annotate(n=Count("id"))}
    top_doctors = (
        Doctor.objects.annotate(booking_count=Count("consultations"))
        .filter(booking_count__gt=0)
        .order_by("-booking_count")[:5]
    )

    context = {
        "new_requests": new_requests[:10],
        "new_count": new_requests.count(),
        "recent": consultations[:8],
        "doctor_count": Doctor.objects.count(),
        "active_doctor_count": Doctor.objects.filter(is_active=True).count(),
        "total_consultations": consultations.count(),
        "pending_count": counts.get(Consultation.STATUS_PENDING, 0),
        "confirmed_count": counts.get(Consultation.STATUS_CONFIRMED, 0),
        "completed_count": counts.get(Consultation.STATUS_COMPLETED, 0),
        "cancelled_count": counts.get(Consultation.STATUS_CANCELLED, 0),
        "today_count": consultations.filter(preferred_date=date.today()).count(),
        "top_doctors": top_doctors,
    }
    return render(request, "consultations/admin/dashboard.html", context)


@staff_required
def admin_doctor_list_view(request):
    doctors = Doctor.objects.annotate(booking_count=Count("consultations"))
    query = request.GET.get("q", "").strip()
    status = request.GET.get("status", "").strip()
    if query:
        doctors = doctors.filter(
            Q(full_name__icontains=query) | Q(city__icontains=query) | Q(specialization__icontains=query)
        )
    if status == "active":
        doctors = doctors.filter(is_active=True)
    elif status == "inactive":
        doctors = doctors.filter(is_active=False)

    return render(request, "consultations/admin/doctor_list.html", {
        "doctors": doctors,
        "selected": {"q": query, "status": status},
    })


@staff_required
def admin_doctor_create_view(request):
    form = DoctorForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        doctor = form.save()
        messages.success(request, f"{doctor.display_name} has been added to the directory.")
        return redirect("manage:doctors")
    return render(request, "consultations/admin/doctor_form.html", {"form": form, "doctor": None})


@staff_required
def admin_doctor_edit_view(request, pk):
    doctor = get_object_or_404(Doctor, pk=pk)
    form = DoctorForm(request.POST or None, request.FILES or None, instance=doctor)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, f"{doctor.display_name} profile has been updated.")
        return redirect("manage:doctors")
    return render(request, "consultations/admin/doctor_form.html", {"form": form, "doctor": doctor})


@staff_required
def admin_doctor_toggle_view(request, pk):
    doctor = get_object_or_404(Doctor, pk=pk)
    if request.method == "POST":
        doctor.is_active = not doctor.is_active
        doctor.save(update_fields=["is_active", "updated_at"])
        state = "listed to users" if doctor.is_active else "hidden from users"
        messages.info(request, f"{doctor.display_name} is now {state}.")
    return redirect("manage:doctors")


@staff_required
def admin_doctor_delete_view(request, pk):
    doctor = get_object_or_404(Doctor, pk=pk)
    if request.method == "POST":
        if doctor.consultations.exists():
            messages.error(
                request,
                f"{doctor.display_name} has existing consultations and cannot be deleted. "
                "Deactivate the profile instead.",
            )
        else:
            name = doctor.display_name
            doctor.delete()
            messages.success(request, f"{name} has been removed from the directory.")
    return redirect("manage:doctors")


@staff_required
def admin_consultation_list_view(request):
    consultations = Consultation.objects.select_related("doctor", "user")
    status = request.GET.get("status", "").strip()
    query = request.GET.get("q", "").strip()
    only_new = request.GET.get("new") == "1"

    if status:
        consultations = consultations.filter(status=status)
    if only_new:
        consultations = consultations.filter(is_seen_by_admin=False)
    if query:
        consultations = consultations.filter(
            Q(pet_name__icontains=query)
            | Q(owner_name__icontains=query)
            | Q(doctor__full_name__icontains=query)
            | Q(predicted_display_name__icontains=query)
        )

    return render(request, "consultations/admin/consultation_list.html", {
        "consultations": consultations,
        "statuses": Consultation.STATUS_CHOICES,
        "selected": {"status": status, "q": query, "new": only_new},
        "new_count": Consultation.objects.filter(is_seen_by_admin=False).count(),
    })


@staff_required
def admin_consultation_detail_view(request, pk):
    consultation = get_object_or_404(Consultation.objects.select_related("doctor", "user"), pk=pk)
    was_new = not consultation.is_seen_by_admin

    if request.method == "POST":
        form = ConsultationStatusForm(request.POST, instance=consultation)
        if form.is_valid():
            updated = form.save(commit=False)
            updated.is_seen_by_admin = True
            updated.save()
            messages.success(request, f"Consultation marked as {updated.get_status_display()}.")
            return redirect("manage:consultations")
        messages.error(request, "Please correct the highlighted fields.")
    else:
        form = ConsultationStatusForm(instance=consultation)
        if was_new:
            # Opening the request clears the highlighted notification for it.
            consultation.is_seen_by_admin = True
            consultation.save(update_fields=["is_seen_by_admin", "updated_at"])

    return render(request, "consultations/admin/consultation_detail.html", {
        "consultation": consultation,
        "form": form,
        "was_new": was_new,
        "now": timezone.now(),
    })


@staff_required
def admin_mark_all_seen_view(request):
    if request.method == "POST":
        updated = Consultation.objects.filter(is_seen_by_admin=False).update(is_seen_by_admin=True)
        messages.info(request, f"{updated} notification(s) cleared.")
    return redirect(request.POST.get("next") or "manage:dashboard")
