import os
import uuid

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render
from PIL import Image

from dog_disease_project.data_loader import get_disease_info

from .model_service import predict_image

SAMPLES_DIR = settings.BASE_DIR / "static" / "images" / "samples"


def _list_sample_images():
    samples = []
    if SAMPLES_DIR.exists():
        for cls_dir in sorted(SAMPLES_DIR.iterdir()):
            if cls_dir.is_dir():
                for f in sorted(cls_dir.iterdir())[:4]:
                    samples.append({
                        "class_name": cls_dir.name,
                        "url": f"images/samples/{cls_dir.name}/{f.name}",
                        "relative_path": f"{cls_dir.name}/{f.name}",
                    })
    return samples


def _severity_badge(severity_text):
    text = (severity_text or "").lower()
    if "severe" in text:
        return "severe"
    if "moderate" in text:
        return "moderate"
    if "none" in text:
        return "none"
    if "mild" in text:
        return "mild"
    return "moderate"


def _build_result(ranked, image_url):
    top1_name, top1_prob = ranked[0]
    disease_info = get_disease_info().get(top1_name, {})
    return {
        "image_url": image_url,
        "predicted_class": top1_name,
        "display_name": disease_info.get("display_name", top1_name),
        "confidence": round(top1_prob * 100, 1),
        "top3": [
            {"class_name": name, "confidence": round(prob * 100, 1)}
            for name, prob in ranked[:3]
        ],
        "info": disease_info,
        "severity_badge": _severity_badge(disease_info.get("severity")),
    }


def _remember_prediction(request, result):
    """Stash the screening result so the consultation booking form can default to it."""
    info = result.get("info") or {}
    request.session["last_prediction"] = {
        "predicted_class": result["predicted_class"],
        "display_name": result["display_name"],
        "confidence": result["confidence"],
        "severity": info.get("severity", ""),
        "description": info.get("description", ""),
        "symptoms": info.get("symptoms", [])[:4],
        "image_url": result["image_url"],
    }


@login_required
def prediction_view(request):
    result = None
    samples = _list_sample_images()

    if request.method == "POST":
        try:
            if "image" in request.FILES and request.FILES["image"]:
                upload = request.FILES["image"]
                if upload.size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    messages.error(request, f"Image must be smaller than {settings.MAX_UPLOAD_SIZE_MB} MB.")
                    return render(request, "prediction.html", {"result": None, "samples": samples})

                ext = os.path.splitext(upload.name)[1] or ".jpg"
                filename = f"{uuid.uuid4().hex}{ext}"
                upload_dir = settings.MEDIA_ROOT / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                dest_path = upload_dir / filename
                with open(dest_path, "wb") as f:
                    for chunk in upload.chunks():
                        f.write(chunk)

                pil_image = Image.open(dest_path)
                ranked = predict_image(pil_image)
                image_url = f"{settings.MEDIA_URL}uploads/{filename}"
                result = _build_result(ranked, image_url)
                _remember_prediction(request, result)

            elif request.POST.get("sample_image"):
                rel_path = request.POST["sample_image"]
                full_path = (SAMPLES_DIR / rel_path).resolve()
                samples_root = SAMPLES_DIR.resolve()
                if samples_root not in full_path.parents or not full_path.exists():
                    messages.error(request, "Selected sample image could not be found.")
                else:
                    pil_image = Image.open(full_path)
                    ranked = predict_image(pil_image)
                    image_url = f"{settings.STATIC_URL}images/samples/{rel_path}"
                    result = _build_result(ranked, image_url)
                    _remember_prediction(request, result)
            else:
                messages.error(request, "Please upload an image or choose a sample test image.")
        except Exception as exc:  # keep the demo resilient to bad uploads
            messages.error(request, f"Could not process this image: {exc}")

    return render(request, "prediction.html", {"result": result, "samples": samples})
