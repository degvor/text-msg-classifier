from django.contrib import admin
from .models import StreamStatus, ClassifiedMessage

# Register your models here.

admin.site.register(StreamStatus)
admin.site.register(ClassifiedMessage)
