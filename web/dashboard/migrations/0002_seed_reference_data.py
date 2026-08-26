"""Seed the label set and the two singleton control rows.

Categories are inserted from ``streamcls.labels``, which is the single source
of truth for class ids. They are deliberately *not* auto-generated from
training data: deriving them from a fitted ``StringIndexer`` is what previously
let a retrain renumber every class and silently change the meaning of every
stored prediction.
"""

from django.db import migrations

from streamcls.labels import LABELS

# Default palette. Chosen to stay distinguishable in both light and dark UI and
# for the two most-confused classes not to be near-neighbours in hue.
_COLORS = {
    "world": "#4c78a8",
    "sports": "#54a24b",
    "business": "#e45756",
    "sci_tech": "#b279a2",
}


def seed(apps, schema_editor):
    Category = apps.get_model("dashboard", "Category")
    StreamConfig = apps.get_model("dashboard", "StreamConfig")
    StreamStatus = apps.get_model("dashboard", "StreamStatus")

    for label in LABELS:
        Category.objects.update_or_create(
            id=label.id,
            defaults={
                "key": label.key,
                "display_name": label.display_name,
                "description": label.description,
                "color": _COLORS.get(label.key, "#8899aa"),
            },
        )

    StreamConfig.objects.get_or_create(pk=1)
    StreamStatus.objects.get_or_create(pk=1, defaults={"is_active": False})


def unseed(apps, schema_editor):
    apps.get_model("dashboard", "Category").objects.filter(
        id__in=[label.id for label in LABELS]
    ).delete()


class Migration(migrations.Migration):
    dependencies = [("dashboard", "0001_initial")]
    operations = [migrations.RunPython(seed, unseed)]
