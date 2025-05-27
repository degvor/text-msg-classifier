from django.db import models

# Create your models here.

class StreamStatus(models.Model):
    is_active = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

class ClassifiedMessage(models.Model):
    post_id = models.CharField(max_length=100, unique=True)
    post_text = models.TextField()
    tokens = models.TextField()
    #tokens = models.JSONField()
    channel_title = models.CharField(max_length=255)
    channel_id = models.CharField(max_length=100)
    post_view_count = models.IntegerField()
    post_created_at = models.DateTimeField()
    category = models.IntegerField()
    status = models.CharField(max_length=50)
    processing_level = models.CharField(max_length=50)
    parsed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.post_id} - {self.category}"
