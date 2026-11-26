from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme

from .forms import ForgotPasswordForm, LoginForm, RegisterForm
from .models import Profile


def _safe_next(request):
    """Return the ?next= target only when it points back at this site."""
    target = request.POST.get("next") or request.GET.get("next")
    if target and url_has_allowed_host_and_scheme(
        target, allowed_hosts={request.get_host()}, require_https=request.is_secure()
    ):
        return target
    return None


def login_view(request):
    """Single sign-in page with two modes: pet owner and admin portal."""
    if request.user.is_authenticated:
        return redirect("dashboard:home")

    role = (request.POST.get("role") or request.GET.get("role") or "user").strip().lower()
    if role != "admin":
        role = "user"

    form = LoginForm(request.POST or None, initial={"role": role})
    if request.method == "POST" and form.is_valid():
        identifier = form.cleaned_data["identifier"].strip()
        password = form.cleaned_data["password"]
        remember_me = form.cleaned_data["remember_me"]
        role = form.cleaned_data["role"]

        username = identifier
        if "@" in identifier:
            try:
                username = User.objects.get(email__iexact=identifier).username
            except User.DoesNotExist:
                username = identifier

        user = authenticate(request, username=username, password=password)
        if user is None:
            messages.error(request, "Invalid email/username or password.")
        elif role == "admin" and not user.is_staff:
            # Correct credentials, but this account has no admin portal access.
            messages.error(
                request,
                "This account does not have admin access. Sign in through the Pet Owner tab instead.",
            )
        else:
            login(request, user)
            request.session.set_expiry(60 * 60 * 24 * 14 if remember_me else 0)

            if role == "admin":
                messages.success(request, f"Signed in to the admin portal as {user.username}.")
                destination = _safe_next(request) or reverse("manage:dashboard")
            else:
                messages.success(request, f"Welcome back, {user.first_name or user.username}!")
                destination = _safe_next(request) or reverse("dashboard:home")
            return redirect(destination)

    return render(request, "login.html", {
        "form": form,
        "role": role,
        "next": _safe_next(request) or "",
    })


def register_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard:home")

    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data
        user = User.objects.create_user(
            username=data["username"],
            email=data["email"],
            password=data["password"],
            first_name=data["full_name"].split(" ")[0],
        )
        Profile.objects.create(user=user, full_name=data["full_name"])
        messages.success(request, "Account created successfully. Please sign in.")
        return redirect("accounts:login")

    return render(request, "register.html", {"form": form})


def logout_view(request):
    logout(request)
    messages.info(request, "You have been signed out.")
    return redirect("accounts:login")


def forgot_password_view(request):
    form = ForgotPasswordForm(request.POST or None)
    submitted = False
    if request.method == "POST" and form.is_valid():
        # Demo/UI-only flow: no real email backend is configured.
        submitted = True
        messages.info(
            request,
            "If an account exists for that email, password reset instructions have been sent.",
        )
    return render(request, "forgot_password.html", {"form": form, "submitted": submitted})
