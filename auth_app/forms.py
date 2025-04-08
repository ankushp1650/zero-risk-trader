from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm


class RegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']


class ApiKeyForm(forms.Form):
    api_key = forms.CharField(max_length=255, widget=forms.TextInput(attrs={'placeholder': 'Enter your API key'}))


class AlphaVantageRegistrationForm(forms.Form):
    email = forms.EmailField(label='Your Email', required=True)
