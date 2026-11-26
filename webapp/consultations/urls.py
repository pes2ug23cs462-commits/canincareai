from django.urls import path

from . import views

app_name = "consultations"

urlpatterns = [
    # User-facing
    path("", views.doctor_list_view, name="doctors"),
    path("doctor/<int:pk>/", views.doctor_detail_view, name="doctor_detail"),
    path("doctor/<int:pk>/book/", views.book_consultation_view, name="book"),
    path("my-bookings/", views.my_bookings_view, name="my_bookings"),
    path("my-bookings/<int:pk>/cancel/", views.cancel_booking_view, name="cancel_booking"),
]
