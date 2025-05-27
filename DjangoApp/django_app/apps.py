from django.apps import AppConfig


class DjangoAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'django_app'

    def ready(self):
        # from django.db.utils import OperationalError
        # try:
        #     from .models import StreamStatus
        #     if StreamStatus.objects.count() == 0:
        #         StreamStatus.objects.create(is_active=False, updated_at=None)
        # except OperationalError:
        #     pass
        import django_app.signals
