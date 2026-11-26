import json

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST

from .geoapify_service import reverse_geocode, forward_geocode


@login_required
def clinics_view(request):
    return render(request, "clinics.html")


@login_required
@require_POST
@csrf_protect
def location_api(request):
    """Resolve the browser's reported coordinates to an address (Geoapify),
    and build a Google Maps search link for nearby pet clinics there."""
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid request."}, status=400)

    lat = payload.get("latitude")
    lng = payload.get("longitude")
    if lat is None or lng is None:
        return JsonResponse({"error": "Latitude and longitude are required."}, status=400)

    geocoded = reverse_geocode(lat, lng)
    address = geocoded["formatted"] if geocoded else "Not available"

    maps_url = f"https://www.google.com/maps/search/pet+clinic/@{lat},{lng},14z"

    return JsonResponse({
        "latitude": lat,
        "longitude": lng,
        "address": address,
        "google_maps": maps_url,
    })


@login_required
@require_POST
@csrf_protect
def area_api(request):
    """Turn a typed area / city name into coordinates and a clinic search link,
    so users don't have to know their latitude and longitude."""
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid request."}, status=400)

    area = (payload.get("area") or "").strip()
    if not area:
        return JsonResponse({"error": "Please enter an area or city."}, status=400)

    geo = forward_geocode(area)
    if not geo:
        return JsonResponse(
            {"error": "Could not find that place. Try adding the state, or a nearby city."},
            status=404,
        )

    lat, lng = geo["lat"], geo["lng"]
    maps_url = f"https://www.google.com/maps/search/pet+clinic/@{lat},{lng},14z"

    return JsonResponse({
        "latitude": lat,
        "longitude": lng,
        "address": geo["formatted"],
        "google_maps": maps_url,
    })
