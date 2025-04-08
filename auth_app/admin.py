from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from . import models
from .models import  Portfolio, Transaction, UserProfile

# admin.site.register(Stock)
admin.site.register(Portfolio)
admin.site.register(Transaction)
admin.site.register(UserProfile)
