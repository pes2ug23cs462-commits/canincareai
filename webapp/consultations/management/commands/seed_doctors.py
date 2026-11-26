"""Seed a starter set of veterinary doctors so the directory is not empty.

Usage:  python manage.py seed_doctors
"""
from datetime import time

from django.core.management.base import BaseCommand

from consultations.models import Doctor

DOCTORS = [
    {
        "full_name": "Ananya Raghavan",
        "specialization": "Veterinary Dermatology",
        "qualification": "BVSc & AH, MVSc (Dermatology)",
        "experience_years": 18,
        "clinic_name": "PawCare Skin & Allergy Clinic",
        "city": "Bengaluru",
        "address": "24, 4th Cross, Indiranagar, Bengaluru 560038",
        "consultation_mode": Doctor.CONSULT_BOTH,
        "consultation_fee": 800,
        "languages": "English, Kannada, Hindi",
        "available_days": "Mon - Sat",
        "available_from": time(10, 0),
        "available_to": time(18, 0),
        "phone": "+91 98450 11223",
        "email": "ananya@pawcare.example",
        "bio": "Specialises in chronic canine skin conditions, allergy testing and long-term dermatitis management.",
    },
    {
        "full_name": "Rahul Menon",
        "specialization": "Small Animal Medicine",
        "qualification": "BVSc & AH",
        "experience_years": 12,
        "clinic_name": "Happy Tails Veterinary Hospital",
        "city": "Kochi",
        "address": "Panampilly Nagar, Kochi 682036",
        "consultation_mode": Doctor.CONSULT_CLINIC,
        "consultation_fee": 600,
        "languages": "English, Malayalam",
        "available_days": "Mon - Fri",
        "available_from": time(9, 30),
        "available_to": time(17, 0),
        "phone": "+91 98470 55667",
        "email": "rahul@happytails.example",
        "bio": "General small animal practice with a focus on parasitic skin infections such as mange and demodicosis.",
    },
    {
        "full_name": "Sneha Kulkarni",
        "specialization": "Veterinary Dermatology",
        "qualification": "BVSc, PGDVD",
        "experience_years": 8,
        "clinic_name": "SkinVet Care",
        "city": "Pune",
        "address": "Baner Road, Pune 411045",
        "consultation_mode": Doctor.CONSULT_ONLINE,
        "consultation_fee": 450,
        "languages": "English, Marathi, Hindi",
        "available_days": "Tue - Sun",
        "available_from": time(11, 0),
        "available_to": time(20, 0),
        "phone": "+91 99220 44556",
        "email": "sneha@skinvet.example",
        "bio": "Runs teleconsultations for fungal infections and ringworm, with follow-up photo reviews.",
    },
    {
        "full_name": "Imran Sheikh",
        "specialization": "Veterinary Surgery & Dermatology",
        "qualification": "BVSc & AH, MVSc (Surgery)",
        "experience_years": 21,
        "clinic_name": "City Pet Hospital",
        "city": "Hyderabad",
        "address": "Road No. 12, Banjara Hills, Hyderabad 500034",
        "consultation_mode": Doctor.CONSULT_BOTH,
        "consultation_fee": 1000,
        "languages": "English, Telugu, Urdu, Hindi",
        "available_days": "Mon - Sat",
        "available_from": time(10, 30),
        "available_to": time(19, 0),
        "phone": "+91 90000 77889",
        "email": "imran@citypet.example",
        "bio": "Two decades of practice covering severe hypersensitivity cases and surgical dermatology.",
    },
    {
        "full_name": "Priya Nair",
        "specialization": "Canine Nutrition & Skin Health",
        "qualification": "BVSc, MSc (Animal Nutrition)",
        "experience_years": 6,
        "clinic_name": "Wellness Vet Studio",
        "city": "Chennai",
        "address": "Adyar, Chennai 600020",
        "consultation_mode": Doctor.CONSULT_BOTH,
        "consultation_fee": 500,
        "languages": "English, Tamil",
        "available_days": "Wed - Sun",
        "available_from": time(12, 0),
        "available_to": time(19, 30),
        "phone": "+91 94440 33221",
        "email": "priya@wellnessvet.example",
        "bio": "Works on diet-driven skin recovery plans alongside medical treatment for recurring dermatitis.",
    },
    {
        "full_name": "Vikram Desai",
        "specialization": "Veterinary Dermatology",
        "qualification": "BVSc & AH, MVSc",
        "experience_years": 15,
        "clinic_name": "Furry Friends Clinic",
        "city": "Mumbai",
        "address": "Linking Road, Bandra West, Mumbai 400050",
        "consultation_mode": Doctor.CONSULT_CLINIC,
        "consultation_fee": 900,
        "languages": "English, Hindi, Marathi, Gujarati",
        "available_days": "Mon - Sat",
        "available_from": time(9, 0),
        "available_to": time(16, 0),
        "phone": "+91 98200 66778",
        "email": "vikram@furryfriends.example",
        "bio": "Handles referral dermatology cases, skin biopsies and resistant fungal infections.",
    },
]


class Command(BaseCommand):
    help = "Create a starter set of veterinary doctors for the consultation directory."

    def handle(self, *args, **options):
        created = updated = 0
        for entry in DOCTORS:
            _, was_created = Doctor.objects.update_or_create(
                full_name=entry["full_name"], city=entry["city"], defaults=entry
            )
            created += was_created
            updated += not was_created
        self.stdout.write(self.style.SUCCESS(f"Doctors created: {created}, updated: {updated}"))
