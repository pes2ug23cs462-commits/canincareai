from django.contrib.auth.models import User
from django import forms


class LoginForm(forms.Form):
    identifier = forms.CharField(
        label="Email or Username",
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "you@example.com or username",
            "autofocus": True,
        }),
    )
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": "Enter your password",
        }),
    )
    remember_me = forms.BooleanField(required=False, initial=True, label="Remember me")
    # Which portal the visitor is signing in to: the pet-owner app or the admin portal.
    role = forms.ChoiceField(
        choices=[("user", "Pet Owner"), ("admin", "Admin")],
        required=False,
        initial="user",
        widget=forms.HiddenInput(),
    )

    def clean_role(self):
        return self.cleaned_data.get("role") or "user"


class RegisterForm(forms.Form):
    full_name = forms.CharField(
        label="Full Name",
        max_length=150,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Jane Doe"}),
    )
    email = forms.EmailField(
        label="Email",
        widget=forms.EmailInput(attrs={"class": "form-control", "placeholder": "you@example.com"}),
    )
    username = forms.CharField(
        label="Username",
        max_length=150,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Choose a username"}),
    )
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={"class": "form-control", "placeholder": "Create a password"}),
    )
    confirm_password = forms.CharField(
        label="Confirm Password",
        widget=forms.PasswordInput(attrs={"class": "form-control", "placeholder": "Re-enter your password"}),
    )

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError("This username is already taken.")
        return username

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email

    def clean(self):
        cleaned = super().clean()
        password = cleaned.get("password")
        confirm_password = cleaned.get("confirm_password")
        if password and confirm_password and password != confirm_password:
            self.add_error("confirm_password", "Passwords do not match.")
        if password and len(password) < 8:
            self.add_error("password", "Password must be at least 8 characters long.")
        return cleaned


class ForgotPasswordForm(forms.Form):
    email = forms.EmailField(
        label="Email",
        widget=forms.EmailInput(attrs={"class": "form-control", "placeholder": "you@example.com"}),
    )
