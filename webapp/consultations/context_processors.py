"""Makes the admin portal's consultation notification available site-wide."""
from .models import Consultation


def admin_notifications(request):
    user = getattr(request, "user", None)
    if not (user and user.is_authenticated and user.is_staff):
        return {}

    new_qs = Consultation.objects.filter(is_seen_by_admin=False).select_related("doctor")
    return {
        "admin_new_consultations": new_qs[:5],
        "admin_new_consultation_count": new_qs.count(),
    }
