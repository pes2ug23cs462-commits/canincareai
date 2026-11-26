"""Geoapify reverse-geocoding: turns raw coordinates into a readable address
for the Nearby Pet Clinics page.

Docs: https://apidocs.geoapify.com/docs/geocoding/reverse-geocoding/
"""
import requests
from django.conf import settings

REVERSE_GEOCODE_URL = "https://api.geoapify.com/v1/geocode/reverse"
GEOCODE_SEARCH_URL = "https://api.geoapify.com/v1/geocode/search"
REQUEST_TIMEOUT_SEC = 15


class GeocodeError(Exception):
    pass


def reverse_geocode(lat, lng):
    """Resolve coordinates to a human-readable place name.

    Returns a dict with `label` (short "area, city" style name) and
    `formatted` (full address). Returns None if no location could be
    resolved - callers should fall back to the raw coordinates.
    """
    api_key = settings.GEOAPIFY_API_KEY
    if not api_key:
        return None

    try:
        resp = requests.get(
            REVERSE_GEOCODE_URL,
            params={"lat": lat, "lon": lng, "apiKey": api_key, "format": "json"},
            timeout=REQUEST_TIMEOUT_SEC,
        )
    except requests.RequestException:
        return None

    if resp.status_code != 200:
        return None

    results = resp.json().get("results") or []
    if not results:
        return None

    place = results[0]
    area = place.get("suburb") or place.get("neighbourhood") or place.get("district")
    city = place.get("city") or place.get("county")
    state = place.get("state")

    label_parts = [p for p in (area, city, state) if p]
    label = ", ".join(dict.fromkeys(label_parts)) or place.get("formatted", f"{lat}, {lng}")

    return {
        "label": label,
        "formatted": place.get("formatted", label),
        "city": city,
        "state": state,
        "country": place.get("country"),
    }


def forward_geocode(text):
    """Turn a typed place / area / city name into coordinates.

    Returns a dict with `lat`, `lng` and `formatted` (full address),
    or None if the place could not be found. Lets users search by
    place name instead of entering latitude / longitude themselves.
    """
    api_key = settings.GEOAPIFY_API_KEY
    if not api_key or not text:
        return None

    try:
        resp = requests.get(
            GEOCODE_SEARCH_URL,
            params={"text": text, "apiKey": api_key, "format": "json", "limit": 1},
            timeout=REQUEST_TIMEOUT_SEC,
        )
    except requests.RequestException:
        return None

    if resp.status_code != 200:
        return None

    results = resp.json().get("results") or []
    if not results:
        return None

    place = results[0]
    lat = place.get("lat")
    lng = place.get("lon")
    if lat is None or lng is None:
        return None

    return {
        "lat": lat,
        "lng": lng,
        "formatted": place.get("formatted", text),
    }
