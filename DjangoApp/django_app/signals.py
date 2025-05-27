from django.db.models.signals import post_migrate
from django.dispatch import receiver
from .models import StreamStatus

@receiver(post_migrate)
def create_initial_streamstatus(sender, **kwargs):
    if StreamStatus.objects.count() == 0:
        StreamStatus.objects.create(is_active=False, updated_at=None)