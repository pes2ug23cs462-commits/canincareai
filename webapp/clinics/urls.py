from django.urls import path

from . import views

app_name = "clinics"

urlpatterns = [
    path("", views.clinics_view, name="clinics"),
    path("api/location/", views.location_api, name="location_api"),
    path("api/area/", views.area_api, name="area_api"),
]
