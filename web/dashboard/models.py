"""The database schema, which is also the contract between the services.

This replaces a six-file JSON bus on a shared Docker volume. Those files were
read-modify-written by both the Spark driver and Django with no locking, no
atomicity and no owner: every ``open(path, "w")`` left a window where a reader
saw zero bytes, and the dashboard polled into that window every five seconds.

Three conventions matter to code outside Django:

* every field on a table written by raw SQL that has a Python ``default=`` also
  needs a database-level ``db_default=``. Django applies ``default=`` in Python,
  so a psycopg2 INSERT that omits the column hits a NOT NULL violation instead.
  ``batch_metric``, ``classified_message``, ``drift_event``, ``training_job``,
  ``model_version``, ``stream_status`` and (by the matrix runner)
  ``experiment_run`` are all written this way;

* every model pins ``Meta.db_table`` from :class:`streamcls.db.tables.Table`,
  because the Spark and psycopg2 paths address these tables by name and must
  not break when the Django app is renamed;
* ``ClassifiedMessage`` and ``BatchMetric`` are keyed by *(experiment_run, ...)*,
  never by the natural key alone -- the same held-out split is replayed for
  every run.
"""

from __future__ import annotations

from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator, RegexValidator
from django.db import models
from django.db.models import F, Q, Value

from streamcls.db.tables import Table

HEX_COLOR = RegexValidator(
    r"^#[0-9a-fA-F]{6}$",
    "Must be a 6-digit hex colour such as #1f77b4.",
)


def default_actionable_detectors() -> list[str]:
    """Mirror of ``streamcls.drift.monitor.DEFAULT_ACTIONABLE``: error-rate ADWIN
    is the only detector trusted to trigger a retrain by default."""
    return ["adwin_error"]


def validate_detector_list(value) -> None:
    """``actionable_detectors`` must be a list drawn from :class:`Detector`."""
    if not isinstance(value, list):
        raise ValidationError("Must be a list of detector names.")
    unknown = sorted(set(value) - set(Detector.values))
    if unknown:
        raise ValidationError(f"Unknown detectors: {', '.join(unknown)}.")


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
class Category(models.Model):
    """Display metadata for a class id.

    The id is *not* generated here: it is pinned in ``streamcls.labels`` and
    seeded by a data migration. Previously the integer came from a Spark
    ``StringIndexer`` ordered by class frequency, so retraining on differently
    balanced data renumbered every class and silently changed the meaning of
    every stored prediction.
    """

    id = models.IntegerField(primary_key=True)
    key = models.SlugField(max_length=32, unique=True)
    display_name = models.CharField(max_length=64)
    description = models.TextField(blank=True, default="")
    # Validated, because this value is interpolated into the dashboard's CSS.
    # The previous version stored it verbatim from an unauthenticated endpoint.
    color = models.CharField(max_length=7, default="#8899aa", validators=[HEX_COLOR])

    class Meta:
        db_table = Table.CATEGORY
        ordering = ["id"]
        verbose_name_plural = "categories"

    def __str__(self) -> str:
        return f"{self.id} · {self.display_name}"


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
class StreamMode(models.TextChoices):
    #: Producer writes the whole run to Kafka up front; the consumer drains it
    #: with a fixed maxOffsetsPerTrigger, so every batch holds exactly N
    #: messages in offset order and batch composition is deterministic.
    #: All reported experiments use this.
    REPLAY = "replay", "Replay (deterministic batches)"
    #: Rate-limited with jitter. For demos and screenshots only.
    LIVE = "live", "Live (rate-limited)"


class RunStatus(models.TextChoices):
    DRAFT = "draft", "Draft"
    RUNNING = "running", "Running"
    COMPLETED = "completed", "Completed"
    ABORTED = "aborted", "Aborted"


class ExperimentRun(models.Model):
    """One reproducible pass of the stream. The spine of the results schema.

    Every fact the thesis reports is grouped by a run, and the headline figure
    is two runs with the same ``seed`` differing only in ``adaptation_enabled``.

    Rows are inserted by the matrix runner over psycopg2 as well as by Django,
    so every defaulted field carries a ``db_default`` (see the module docstring).
    """

    name = models.CharField(max_length=120)
    notes = models.TextField(blank=True, default="", db_default="")

    #: Which named drift scenario produced ``drift_schedule``; the grouping key
    #: for the results matrix. Free text so ad-hoc runs can leave it blank.
    scenario = models.CharField(max_length=64, blank=True, default="", db_default="")

    seed = models.IntegerField(
        help_text="Same seed + same drift_schedule reproduces the stream exactly."
    )
    mode = models.CharField(
        max_length=16, choices=StreamMode, default=StreamMode.REPLAY, db_default=StreamMode.REPLAY
    )
    status = models.CharField(
        max_length=16, choices=RunStatus, default=RunStatus.DRAFT, db_default=RunStatus.DRAFT
    )

    #: The A/B switch behind the central figure.
    adaptation_enabled = models.BooleanField(default=True, db_default=True)

    total_messages = models.IntegerField(
        default=20_000, validators=[MinValueValidator(1)], db_default=20_000
    )
    rate_msgs_per_sec = models.FloatField(
        default=50.0, help_text="Ignored in replay mode.", db_default=50.0
    )

    #: ``[{"at": 4000, "kind": "concept_abrupt", "params": {...}}, ...]``
    #: Also the ground truth for detection delay and false-alarm rate.
    drift_schedule = models.JSONField(
        default=list, blank=True, db_default=Value([], models.JSONField())
    )

    dataset_split = models.CharField(
        max_length=64, default="agnews-stream-holdout-v1", db_default="agnews-stream-holdout-v1"
    )
    base_model_version = models.ForeignKey(
        "ModelVersion", null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )

    producer_config = models.JSONField(
        default=dict, blank=True, db_default=Value({}, models.JSONField())
    )
    #: The matrix runner snapshots the whole ``StreamConfig`` row plus the cell
    #: identity here, so a run's knobs survive later edits to the singleton.
    consumer_config = models.JSONField(
        default=dict, blank=True, db_default=Value({}, models.JSONField())
    )
    git_commit = models.CharField(max_length=40, blank=True, default="", db_default="")

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = Table.EXPERIMENT_RUN
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"#{self.pk} {self.name} (seed={self.seed})"


# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------
class SingletonModel(models.Model):
    """A table constrained to exactly one row, at ``pk=1``.

    ``StreamStatus`` previously had no such constraint, and the two readers
    disagreed about which row was authoritative: Django used ``.first()``
    (lowest id, unordered) while Spark used ``ORDER BY id DESC LIMIT 1``. With
    more than one row the UI and the pipeline silently diverged.
    """

    id = models.PositiveSmallIntegerField(primary_key=True, default=1)

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):  # pragma: no cover - guard only
        raise RuntimeError(f"{type(self).__name__} is a singleton and cannot be deleted.")

    @classmethod
    def load(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class StreamConfig(SingletonModel):
    """Every tunable, in one place, read at one lifetime.

    Replaces ``config.json``. That file was written by Django under the key
    ``max_message_length`` and read by Spark under the key ``max_len``, so the
    length limit silently never applied; and one field was read once at startup
    while another was re-read every batch, with nothing documenting either.
    """

    # --- ingestion -----------------------------------------------------------
    max_message_length = models.IntegerField(default=4_000, validators=[MinValueValidator(1)])
    max_offsets_per_trigger = models.IntegerField(
        default=500,
        validators=[MinValueValidator(1)],
        help_text="Bounds each micro-batch. Required for deterministic batching.",
    )
    trigger_seconds = models.IntegerField(default=10, validators=[MinValueValidator(1)])

    # --- drift detection -----------------------------------------------------
    adwin_delta = models.FloatField(default=0.002)
    psi_threshold = models.FloatField(default=0.2)
    psi_consecutive_batches = models.IntegerField(default=3, validators=[MinValueValidator(1)])
    grace_batches = models.IntegerField(
        default=5,
        validators=[MinValueValidator(0)],
        help_text="Suppress detections for this many batches after a start or model swap.",
    )

    # --- adaptation ----------------------------------------------------------
    adaptation_enabled = models.BooleanField(default=True)
    auto_promote = models.BooleanField(default=True)
    promote_min_delta = models.FloatField(
        default=0.01,
        help_text="Promote only if new accuracy >= active accuracy - this. Stops oscillation.",
    )
    min_batches_between_retrains = models.IntegerField(default=10, validators=[MinValueValidator(0)])
    max_retrains_per_run = models.IntegerField(default=10, validators=[MinValueValidator(0)])
    retrain_window_rows = models.IntegerField(default=20_000, validators=[MinValueValidator(1)])
    retrain_min_new_rows = models.IntegerField(
        default=2_000,
        validators=[MinValueValidator(0)],
        db_default=2_000,
        help_text=(
            "Defer retraining until this many messages have arrived since the change "
            "point. Drift is detected within a batch or two, when only a few hundred "
            "rows of the new concept exist -- retraining on those produces a worse "
            "model and burns one of the run's retrain budget."
        ),
    )
    retrain_base_rows = models.IntegerField(
        default=20_000,
        validators=[MinValueValidator(0)],
        help_text="Rows sampled from the base corpus to resist catastrophic forgetting.",
    )
    label_delay_batches = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Labels for batch b become visible at b+D. Models delayed feedback.",
    )
    #: Which detectors may trigger a retrain. Every detector always records its
    #: signals as drift events; this decides which of them are acted on. An
    #: experiment variable, so it lives here rather than in a module constant.
    actionable_detectors = models.JSONField(
        default=default_actionable_detectors,
        db_default=Value(["adwin_error"], models.JSONField()),
        validators=[validate_detector_list],
        help_text="Detectors whose signals trigger a retrain; the rest are recorded only.",
    )

    # --- adaptation policy variants (controlled experiments) ------------------
    # Every default below reproduces the guarded policy the thesis matrices
    # ran; each knob switches off or replaces exactly one of its parts, so a
    # comparison changes one thing at a time. All are written by the matrix
    # runner through raw SQL, hence db_default on every one.
    retrain_trigger = models.CharField(
        max_length=16,
        choices=[("drift", "drift signal"), ("periodic", "fixed schedule")],
        default="drift",
        db_default="drift",
        help_text="What queues a retrain: an actionable drift signal, or a fixed schedule.",
    )
    periodic_retrain_batches = models.IntegerField(
        default=0,
        db_default=0,
        validators=[MinValueValidator(0)],
        help_text="With the periodic trigger: queue a retrain every this many batches.",
    )
    retain_pending_drift = models.BooleanField(
        default=True,
        db_default=True,
        help_text=(
            "Hold a drift detected before enough new data exists until it does. Off, such "
            "a detection is dropped and only a later signal can trigger a retrain."
        ),
    )
    new_data_origin = models.CharField(
        max_length=16,
        choices=[("change_point", "known change point"), ("detection", "detection batch")],
        default="change_point",
        db_default="change_point",
        help_text=(
            "Where the count of new-concept messages starts: the change point the "
            "generator injected, or the first message of the batch the drift was detected in."
        ),
    )
    retrain_holdout_rows = models.IntegerField(
        default=0,
        db_default=0,
        validators=[MinValueValidator(0)],
        help_text=(
            "Newest rows the promotion holdout is drawn from (every fifth). 0 means "
            "retrain_min_new_rows."
        ),
    )
    promotion_comparison = models.CharField(
        max_length=16,
        choices=[("live", "live accuracy"), ("matched_holdout", "same holdout")],
        default="live",
        db_default="live",
        help_text=(
            "What the candidate's holdout accuracy is compared with: the incumbent's "
            "live accuracy, or the incumbent scored on the same holdout."
        ),
    )
    monitoring_enabled = models.BooleanField(
        default=True,
        db_default=True,
        help_text="Off: classify and store only, no detectors or drift statistics (cost baseline).",
    )

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = Table.STREAM_CONFIG
        constraints = [models.CheckConstraint(condition=Q(id=1), name="stream_config_singleton")]

    def __str__(self) -> str:
        return "Stream configuration"


class StreamStatus(SingletonModel):
    """The control flag, plus liveness and the desired model version."""

    is_active = models.BooleanField(default=False, db_default=False)
    producer_active = models.BooleanField(default=False, db_default=False)

    active_experiment_run = models.ForeignKey(
        ExperimentRun, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    active_model_version = models.ForeignKey(
        "ModelVersion", null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    #: Set by promotion. The consumer's control loop notices the difference from
    #: ``active_model_version`` and swaps. This is the whole hot-reload signal.
    desired_model_version = models.ForeignKey(
        "ModelVersion", null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )

    consumer_heartbeat_at = models.DateTimeField(null=True, blank=True)
    producer_heartbeat_at = models.DateTimeField(null=True, blank=True)
    last_batch_id = models.BigIntegerField(null=True, blank=True)
    #: The run the consumer currently has a live streaming query for; NULL when
    #: no query is running. ``is_active`` is what the operator *wants*; this is
    #: what the consumer *is doing*, and the only reliable "query has stopped"
    #: signal -- heartbeats alone cannot distinguish idle from mid-batch.
    consumer_query_run_id = models.BigIntegerField(null=True, blank=True)
    last_error = models.TextField(blank=True, default="", db_default="")

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = Table.STREAM_STATUS
        constraints = [models.CheckConstraint(condition=Q(id=1), name="stream_status_singleton")]

    def __str__(self) -> str:
        return f"Stream {'ACTIVE' if self.is_active else 'INACTIVE'}"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
class ModelStatus(models.TextChoices):
    TRAINING = "training", "Training"
    READY = "ready", "Ready"
    ACTIVE = "active", "Active"
    RETIRED = "retired", "Retired"
    FAILED = "failed", "Failed"


class ModelVersion(models.Model):
    """A trained artefact plus everything needed to serve and audit it."""

    version = models.CharField(max_length=64, unique=True)
    status = models.CharField(
        max_length=16, choices=ModelStatus, default=ModelStatus.TRAINING, db_default="training"
    )
    artifact_path = models.CharField(max_length=512)
    algorithm = models.CharField(max_length=64, default="logreg_tfidf", db_default="logreg_tfidf")
    hyperparams = models.JSONField(
        default=dict, blank=True, db_default=Value({}, models.JSONField())
    )

    # --- evaluation ----------------------------------------------------------
    #: Accuracy on the *fixed* held-out test set. Comparable across every run
    #: and every model in the thesis, which is what makes it the headline
    #: number -- but see `recent_accuracy` for why it must not gate promotion.
    test_accuracy = models.FloatField(null=True, blank=True)
    test_macro_f1 = models.FloatField(null=True, blank=True)
    #: Accuracy on recent stream data held out from this model's own training
    #: window, or null when there was none (the initial offline model).
    #:
    #: Promotion is gated on *this*, not on `test_accuracy`. Under concept
    #: drift the fixed test set still encodes the pre-drift labelling, so a
    #: correctly adapted model necessarily scores worse on it -- gating on the
    #: stale set would make adaptation structurally impossible, which is
    #: exactly what happened the first time this loop ran end to end.
    recent_accuracy = models.FloatField(null=True, blank=True)
    #: accuracy, per-class P/R/F1, confusion matrix, support.
    metrics = models.JSONField(
        default=dict, blank=True, db_default=Value({}, models.JSONField())
    )
    train_rows = models.IntegerField(null=True, blank=True)

    # --- drift reference -----------------------------------------------------
    vocab_size = models.IntegerField(null=True, blank=True)
    #: Indices of the top-N training terms that PSI/JS bins fold down to.
    #: Vocabulary PSI over all ~2000 bins is noise at realistic batch sizes.
    top_term_indices = models.JSONField(
        default=list, blank=True, db_default=Value([], models.JSONField())
    )
    #: Normalised reference histogram, computed over the training corpus with
    #: this model's vocabulary. Swaps with the model, because PSI is defined
    #: relative to *this* model's training distribution.
    reference_histogram = models.JSONField(
        default=list, blank=True, db_default=Value([], models.JSONField())
    )
    #: The model's own confidence profile on held-out data. Comparing a batch
    #: against a *uniform* reference is meaningless -- a calibrated model
    #: concentrates mass in the top bins, so the PSI is enormous and constant.
    #: Against this, movement means "the model has become less sure than it
    #: normally is", which is a usable label-free drift proxy.
    confidence_reference = models.JSONField(
        default=list, blank=True, db_default=Value([], models.JSONField())
    )

    created_by_run = models.ForeignKey(
        ExperimentRun, null=True, blank=True, on_delete=models.SET_NULL, related_name="models"
    )
    git_commit = models.CharField(max_length=40, blank=True, default="", db_default="")
    trained_at = models.DateTimeField(auto_now_add=True)
    activated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = Table.MODEL_VERSION
        ordering = ["-trained_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["status"],
                condition=Q(status=ModelStatus.ACTIVE),
                name="only_one_active_model_version",
            )
        ]

    def __str__(self) -> str:
        acc = f"{self.test_accuracy:.4f}" if self.test_accuracy is not None else "n/a"
        return f"{self.version} [{self.status}] acc={acc}"


class JobStatus(models.TextChoices):
    QUEUED = "queued", "Queued"
    RUNNING = "running", "Running"
    SUCCEEDED = "succeeded", "Succeeded"
    FAILED = "failed", "Failed"


class TrainingJob(models.Model):
    """Work queue between the consumer (which detects drift) and the trainer.

    The trainer claims rows with ``SELECT ... FOR UPDATE SKIP LOCKED``, which is
    atomic and loses no jobs. The consumer does not launch Spark itself: it
    already holds a driver running a streaming query, and forking a second JVM
    in that container would degrade the very throughput being measured.
    """

    experiment_run = models.ForeignKey(
        ExperimentRun, on_delete=models.CASCADE, related_name="training_jobs"
    )
    trigger_drift_event = models.ForeignKey(
        "DriftEvent", null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    status = models.CharField(
        max_length=16, choices=JobStatus, default=JobStatus.QUEUED, db_default="queued"
    )
    params = models.JSONField(
        default=dict, blank=True, db_default=Value({}, models.JSONField())
    )

    model_version = models.ForeignKey(
        ModelVersion, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    error = models.TextField(blank=True, default="", db_default="")

    requested_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = Table.TRAINING_JOB
        ordering = ["id"]
        indexes = [models.Index(fields=["status", "id"], name="training_job_claim_idx")]

    def __str__(self) -> str:
        return f"job#{self.pk} [{self.status}] run={self.experiment_run_id}"


# ---------------------------------------------------------------------------
# Stream output
# ---------------------------------------------------------------------------
class ClassifiedMessage(models.Model):
    """One classified post. Replaces the append-only ``all_messages.json``.

    That file was full-scanned by four dashboard endpoints on a five-second
    poll, parsing the entire corpus into memory to return ten rows.
    """

    experiment_run = models.ForeignKey(
        ExperimentRun, on_delete=models.CASCADE, related_name="messages"
    )
    post_id = models.CharField(max_length=64)
    msg_index = models.IntegerField()

    post_text = models.TextField()
    channel_title = models.CharField(max_length=255, blank=True, default="", db_default="")
    channel_id = models.CharField(max_length=64, blank=True, default="", db_default="")
    view_count = models.IntegerField(default=0, db_default=0)

    true_label = models.ForeignKey(
        Category, on_delete=models.PROTECT, related_name="+", db_column="true_label_id"
    )
    predicted_label = models.ForeignKey(
        Category, on_delete=models.PROTECT, related_name="+", db_column="predicted_label_id"
    )
    #: Max class probability. The previous consumer selected only ``prediction``
    #: and discarded ``probability``, so there was no confidence signal at all
    #: and no way to abstain.
    confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    probabilities = models.JSONField(default=list, blank=True)
    is_correct = models.BooleanField()

    model_version = models.ForeignKey(
        ModelVersion, on_delete=models.PROTECT, related_name="predictions"
    )
    batch_id = models.BigIntegerField()

    # Ground truth about the injected drift, for evaluation only.
    perturbation_kind = models.CharField(max_length=32, default="none", db_default="none")
    drift_phase = models.IntegerField(default=0, db_default=0)

    produced_at = models.DateTimeField()
    classified_at = models.DateTimeField()

    class Meta:
        db_table = Table.CLASSIFIED_MESSAGE
        constraints = [
            # Run-scoped, NOT a bare post_id. The same held-out split is
            # replayed every run, so a bare unique key would make run 2 either
            # overwrite run 1 or write nothing -- invisibly.
            models.UniqueConstraint(
                fields=["experiment_run", "post_id"], name="classified_message_unique_per_run"
            )
        ]
        indexes = [
            models.Index(fields=["experiment_run", "batch_id"], name="cm_run_batch_idx"),
            models.Index(fields=["experiment_run", "msg_index"], name="cm_run_index_idx"),
            models.Index(fields=["experiment_run", "predicted_label"], name="cm_run_pred_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.post_id} -> {self.predicted_label_id}"


class BatchMetric(models.Model):
    """Per-micro-batch measurements. Replaces two JSON files at once.

    ``stream_metrics.json`` reported ``received`` and ``classified`` as the same
    number, so nothing counted drops, and derived throughput from a timestamp
    the same function overwrote -- measuring the producer's emission rate rather
    than system capacity, which is why status was permanently "slow".
    ``performance_logs.json`` was read-all/append-one/write-all every 30 s
    forever, on the driver's critical path.
    """

    experiment_run = models.ForeignKey(
        ExperimentRun, on_delete=models.CASCADE, related_name="batch_metrics"
    )
    batch_id = models.BigIntegerField()
    model_version = models.ForeignKey(
        ModelVersion, on_delete=models.PROTECT, related_name="batch_metrics"
    )

    started_at = models.DateTimeField()
    # db_default, not just default: Django's `default=` is applied in Python, so
    # a raw-SQL writer that omits the column hits a NOT NULL violation. This
    # table is written by the Spark consumer over psycopg2, not through the ORM,
    # so every nullable-by-intent column needs a database-level default.
    duration_ms = models.IntegerField(default=0, db_default=0)

    # --- volume: counted separately, unlike before ---------------------------
    n_received = models.IntegerField(help_text="Rows read from Kafka, before filtering.")
    n_classified = models.IntegerField(help_text="Rows that survived filtering and were scored.")
    n_correct = models.IntegerField(default=0, db_default=0)
    accuracy = models.FloatField(null=True, blank=True)
    cumulative_accuracy = models.FloatField(null=True, blank=True)

    # --- distributions -------------------------------------------------------
    mean_confidence = models.FloatField(null=True, blank=True)
    std_confidence = models.FloatField(null=True, blank=True)
    confidence_hist = models.JSONField(default=list, blank=True, db_default=Value([], models.JSONField()))
    predicted_hist = models.JSONField(default=list, blank=True, db_default=Value([], models.JSONField()))
    true_hist = models.JSONField(default=list, blank=True, db_default=Value([], models.JSONField()))

    # --- drift statistics ----------------------------------------------------
    psi_vocabulary = models.FloatField(null=True, blank=True)
    psi_confidence = models.FloatField(null=True, blank=True)
    psi_predicted = models.FloatField(null=True, blank=True)
    js_vocabulary = models.FloatField(null=True, blank=True)
    adwin_estimate = models.FloatField(null=True, blank=True)
    adwin_width = models.IntegerField(null=True, blank=True)
    ddm_warning = models.BooleanField(default=False, db_default=False)

    #: One bit per message: was the prediction correct. 500 messages = 63 bytes.
    #: Lets detector state be rebuilt *exactly* by replay after a restart,
    #: instead of pickling river objects whose internals are not a stable
    #: contract.
    correct_bits = models.BinaryField(null=True, blank=True)

    throughput_msgs_s = models.FloatField(null=True, blank=True)
    kafka_lag = models.BigIntegerField(null=True, blank=True)

    class Meta:
        db_table = Table.BATCH_METRIC
        ordering = ["experiment_run", "batch_id"]
        constraints = [
            models.UniqueConstraint(
                fields=["experiment_run", "batch_id"], name="batch_metric_unique_per_run"
            ),
            models.CheckConstraint(
                condition=Q(n_classified__lte=F("n_received")),
                name="batch_metric_classified_le_received",
            ),
        ]

    def __str__(self) -> str:
        return f"run={self.experiment_run_id} batch={self.batch_id} acc={self.accuracy}"


class Detector(models.TextChoices):
    DDM = "ddm", "DDM (error rate)"
    ADWIN_ERROR = "adwin_error", "ADWIN (error rate)"
    ADWIN_CONFIDENCE = "adwin_confidence", "ADWIN (confidence)"
    PSI_VOCABULARY = "psi_vocabulary", "PSI (vocabulary)"
    PSI_CONFIDENCE = "psi_confidence", "PSI (confidence)"
    PSI_PREDICTED = "psi_predicted", "PSI (predicted class)"


class DriftEvent(models.Model):
    """A detector firing, and what happened as a result."""

    experiment_run = models.ForeignKey(
        ExperimentRun, on_delete=models.CASCADE, related_name="drift_events"
    )
    batch_id = models.BigIntegerField()
    detector = models.CharField(max_length=32, choices=Detector)

    statistic = models.FloatField(null=True, blank=True)
    threshold = models.FloatField(null=True, blank=True)
    accuracy_before = models.FloatField(null=True, blank=True)

    #: True when the signal fired inside the post-start / post-swap grace
    #: window. Recorded rather than dropped, so suppression is auditable.
    suppressed = models.BooleanField(default=False, db_default=False)

    #: Nearest preceding injected change point, from the run's drift_schedule.
    #: Detection delay is only computable because the producer ships this.
    ground_truth_drift_index = models.IntegerField(null=True, blank=True)
    detection_delay_batches = models.IntegerField(null=True, blank=True)

    action_taken = models.CharField(max_length=32, blank=True, default="none", db_default="none")
    training_job = models.ForeignKey(
        TrainingJob, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    detected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = Table.DRIFT_EVENT
        ordering = ["experiment_run", "batch_id", "id"]
        indexes = [
            models.Index(fields=["experiment_run", "batch_id"], name="drift_run_batch_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.detector}@batch{self.batch_id} run={self.experiment_run_id}"
