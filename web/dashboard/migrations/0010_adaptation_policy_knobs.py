import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0009_matrix_support'),
    ]

    operations = [
        migrations.AddField(
            model_name='streamconfig',
            name='retrain_trigger',
            field=models.CharField(choices=[('drift', 'drift signal'), ('periodic', 'fixed schedule')], db_default='drift', default='drift', help_text='What queues a retrain: an actionable drift signal, or a fixed schedule.', max_length=16),
        ),
        migrations.AddField(
            model_name='streamconfig',
            name='periodic_retrain_batches',
            field=models.IntegerField(db_default=0, default=0, help_text='With the periodic trigger: queue a retrain every this many batches.', validators=[django.core.validators.MinValueValidator(0)]),
        ),
        migrations.AddField(
            model_name='streamconfig',
            name='retain_pending_drift',
            field=models.BooleanField(db_default=True, default=True, help_text='Hold a drift detected before enough new data exists until it does. Off, such a detection is dropped and only a later signal can trigger a retrain.'),
        ),
        migrations.AddField(
            model_name='streamconfig',
            name='new_data_origin',
            field=models.CharField(choices=[('change_point', 'known change point'), ('detection', 'detection batch')], db_default='change_point', default='change_point', help_text='Where the count of new-concept messages starts: the change point the generator injected, or the first message of the batch the drift was detected in.', max_length=16),
        ),
        migrations.AddField(
            model_name='streamconfig',
            name='retrain_holdout_rows',
            field=models.IntegerField(db_default=0, default=0, help_text='Newest rows the promotion holdout is drawn from (every fifth). 0 means retrain_min_new_rows.', validators=[django.core.validators.MinValueValidator(0)]),
        ),
        migrations.AddField(
            model_name='streamconfig',
            name='promotion_comparison',
            field=models.CharField(choices=[('live', 'live accuracy'), ('matched_holdout', 'same holdout')], db_default='live', default='live', help_text="What the candidate's holdout accuracy is compared with: the incumbent's live accuracy, or the incumbent scored on the same holdout.", max_length=16),
        ),
        migrations.AddField(
            model_name='streamconfig',
            name='monitoring_enabled',
            field=models.BooleanField(db_default=True, default=True, help_text='Off: classify and store only, no detectors or drift statistics (cost baseline).'),
        ),
    ]
