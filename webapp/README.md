# CanineCare AI

A Django web application for AI-assisted dog skin disease screening, built on top of the
Stage 1 dataset analysis and model training (see `../notebooks/`).

## Setup

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser   # optional, for /admin/
python manage.py runserver
```

Then open http://127.0.0.1:8000/ and register an account.

## Configuration (`.env`)

Copy `.env.example` to `.env` and fill in:

- `SECRET_KEY` - Django secret key (a default dev key is provided, replace for production)
- `GEMINI_API_KEY` - required for the "AI Assistant" chatbot page. Get a free key at
  https://aistudio.google.com/app/apikey. Without it, the chatbot page still loads but shows a
  clear "not configured" message instead of failing.
- `GEOAPIFY_API_KEY` - used by the "Pet Clinics" page to resolve the user's coordinates to a
  readable address. Get a key at https://www.geoapify.com. Without it, the page still works but
  shows "Not available" for the address.

**Note on the model:** Stage 1 was trained with PyTorch (not TensorFlow/Keras). The app loads
`models/dog_disease_proposed_model.pt` (the native PyTorch weights) for real inference.
`models/dog_disease_proposed_model.h5` is also included as a plain HDF5 export of the same
weights to match the requested filename, but it is not a Keras model and is not used at runtime.

## Doctors & consultations

Two connected surfaces, added on top of the screening flow.

**Admin portal (`/manage/`, staff accounts only)**

- `/manage/` - overview: doctor and consultation counts, status breakdown, most-booked doctors.
- `/manage/doctors/` - add, edit, hide/show and delete doctor profiles. A doctor with existing
  consultations cannot be deleted, only deactivated.
- `/manage/consultations/` - every booking, filterable by status, keyword and "unread only".
- A **highlighted notification** (orange banner + navbar bell badge + pulsing row markers) appears
  whenever a consultation is waiting for review. Opening a request clears its notification;
  "Clear" on the banner clears them all. A user cancelling a booking re-raises the notification.

**Signing in as an admin**

`/accounts/login/` has two modes, switched by the *Pet Owner* / *Admin* tabs on the sign-in card
(`/accounts/login/?role=admin` opens straight in admin mode). The admin tab authenticates the same
credentials but additionally requires `is_staff`; a valid non-staff account is refused with a clear
message and is not signed in. A successful admin sign-in lands on `/manage/` instead of the user
dashboard, and any `?next=` target is honoured (same-site targets only). Visiting a `/manage/` page
while signed out redirects to the login page with the Admin tab already selected.

Grant a user access with `python manage.py createsuperuser`, or by setting `is_staff` on an
existing account.

Demo admin credentials (development only - change or remove before deploying):

- Username: `clinicadmin`
- Password: `TestPass123`

**User side (`/consult/`)**

- Browse active doctors filtered by **years of experience** and **place** (city), plus
  consultation mode, keyword search and sorting by experience, fee or name.
- Booking form fields default to the **last screening result**: the predicted class, its display
  name, confidence, severity, description and symptom list are pulled from the session written by
  the prediction view (`prediction.views._remember_prediction`) and from
  `data/disease_information.json`. The user can override the condition before submitting.
- `/consult/my-bookings/` tracks status (pending / confirmed / completed / cancelled), shows the
  clinic's reply and allows cancellation.

Seed a starter directory of six doctors with:

```bash
python manage.py seed_doctors
```

## What's real vs. simulated

- **Dataset stats, model comparison table, latency**: all measured directly from the Stage 1
  notebooks/scripts - nothing is fabricated.
- **Disease predictions**: real-time inference using the trained PyTorch model.
- **Pet Clinics**: the browser's geolocation coordinates are reverse-geocoded to a real address
  via Geoapify, then a Google Maps "pet clinic" search link centered on that location is opened -
  the address shown is real, resolved data, not a static list.
- **Chatbot**: live calls to the Gemini API with a domain-restricted system prompt, once
  `GEMINI_API_KEY` is set.
