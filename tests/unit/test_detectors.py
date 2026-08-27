"""DDM and ADWIN.

These are the thesis's own implementations, so they are tested on behaviour --
false-alarm rate, detection delay, adaptivity -- rather than against a
reference library. Every stream is seeded, so a failure here is reproducible.
"""

from __future__ import annotations

import random

import pytest

from streamcls.drift.detectors import ADWIN, DDM

DETECTORS = [
    pytest.param(lambda: DDM(), id="ddm"),
    pytest.param(lambda: ADWIN(delta=0.002), id="adwin"),
]


def bernoulli(n: int, p: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [1.0 if rng.random() < p else 0.0 for _ in range(n)]


def count_alarms(detector, stream) -> int:
    alarms = 0
    for value in stream:
        detector.update(value)
        if detector.drift_detected:
            alarms += 1
            detector.reset()
    return alarms


def detection_index(detector, stream) -> int | None:
    for index, value in enumerate(stream):
        detector.update(value)
        if detector.drift_detected:
            return index
    return None


# --- no false alarms ---------------------------------------------------------
@pytest.mark.parametrize("factory", DETECTORS)
@pytest.mark.parametrize("error_rate", [0.05, 0.10, 0.30])
def test_no_alarms_on_a_stationary_stream(factory, error_rate):
    """A detector that fires on noise makes the whole adaptation loop useless:
    it would retrain constantly on data that never changed."""
    assert count_alarms(factory(), bernoulli(4000, error_rate, seed=7)) == 0


@pytest.mark.parametrize("factory", DETECTORS)
def test_no_alarms_on_a_perfect_stream(factory):
    assert count_alarms(factory(), [0.0] * 2000) == 0


def test_adwin_does_not_alarm_when_accuracy_improves():
    """Getting better is not drift worth retraining on.

    Only ADWIN is asserted here. DDM does fire in this situation, for the
    structural reason quantified in :func:`test_false_alarm_rate_is_quantified`
    -- and that difference is a reportable result, not a bug to hide.
    """
    stream = bernoulli(1500, 0.40, seed=3) + bernoulli(1500, 0.05, seed=4)
    detector = ADWIN(delta=0.002)
    detector.extend(stream[:1500])
    detector.reset()
    assert count_alarms(detector, stream[1500:]) == 0


@pytest.mark.parametrize("error_rate", [0.05, 0.10, 0.20, 0.40])
def test_false_alarm_rate_is_quantified(error_rate):
    """DDM fires on stationary noise; ADWIN does not. Measured, not asserted.

    DDM's trigger is ``p + s > p_min + 3 * s_min`` and ``s`` shrinks as the run
    length grows, so on a long stationary stream the band eventually becomes
    tighter than ordinary binomial fluctuation. This test pins the observed
    asymmetry, because it is the reason the adaptation loop needs a cooldown and
    a promotion guard rather than trusting one detector.
    """
    ddm_alarms = adwin_alarms = 0
    for seed in range(40):
        stream = bernoulli(3000, error_rate, seed)
        if detection_index(DDM(), stream) is not None:
            ddm_alarms += 1
        if detection_index(ADWIN(delta=0.002), stream) is not None:
            adwin_alarms += 1
    assert adwin_alarms == 0, "ADWIN should not fire on a stationary stream"
    assert ddm_alarms > 0, "the documented DDM behaviour changed; update the docstring"


# --- detection ---------------------------------------------------------------
@pytest.mark.parametrize("factory", DETECTORS)
def test_detects_an_abrupt_increase_in_error(factory):
    stream = bernoulli(2000, 0.10, seed=11) + bernoulli(2000, 0.40, seed=12)
    index = detection_index(factory(), stream)
    assert index is not None, "no detection at all"
    assert index >= 2000, "fired before the change point"
    assert index - 2000 < 600, f"detection delay {index - 2000} is impractically long"


@pytest.mark.parametrize("factory", DETECTORS)
def test_a_bigger_change_is_detected_sooner(factory):
    """Detection delay should fall as the shift grows; otherwise the delay
    numbers reported in the results chapter mean nothing."""

    def delay(after: float) -> int:
        stream = bernoulli(2000, 0.10, seed=21) + bernoulli(3000, after, seed=22)
        index = detection_index(factory(), stream)
        return (index - 2000) if index is not None else 10**6

    assert delay(0.60) <= delay(0.25)


def test_adwin_detects_a_gradual_drift():
    """DDM is built around abrupt change; ADWIN's windowing is what gives it a
    chance on a ramp."""
    rng = random.Random(5)
    stream = [1.0 if rng.random() < 0.10 else 0.0 for _ in range(1000)]
    stream += [1.0 if rng.random() < (0.10 + 0.5 * i / 2000) else 0.0 for i in range(2000)]
    assert detection_index(ADWIN(delta=0.002), stream) is not None


# --- adaptivity --------------------------------------------------------------
def test_adwin_window_collapses_on_detection():
    """The window shrinking *is* the adaptation: the estimate stops being
    dragged back by data from before the change."""
    stream = bernoulli(2000, 0.10, seed=31) + bernoulli(500, 0.50, seed=32)
    detector = ADWIN(delta=0.002)
    for index, value in enumerate(stream):
        detector.update(value)
        if detector.drift_detected:
            assert index >= 2000
            assert detector.width < 2000
            return
    pytest.fail("no detection")


def test_adwin_memory_is_logarithmic():
    """Exponential-histogram buckets, not raw observations. Without this the
    detector would grow without bound on a long stream."""
    detector = ADWIN()
    detector.extend(bernoulli(50_000, 0.2, seed=41))
    buckets = sum(len(level) for level in detector._levels)
    assert detector.width > 1000
    assert buckets < 100, f"{buckets} buckets for a window of {detector.width}"


def test_adwin_estimation_tracks_the_current_rate():
    detector = ADWIN()
    detector.extend(bernoulli(3000, 0.25, seed=51))
    assert detector.estimation == pytest.approx(0.25, abs=0.05)


# --- state and interface -----------------------------------------------------
@pytest.mark.parametrize("factory", DETECTORS)
def test_reset_clears_state(factory):
    detector = factory()
    detector.extend(bernoulli(1000, 0.5, seed=61))
    detector.reset()
    assert detector.drift_detected is False
    assert detector.statistic == pytest.approx(0.0) or detector.statistic == pytest.approx(1.0)


@pytest.mark.parametrize("factory", DETECTORS)
def test_extend_is_equivalent_to_repeated_update(factory):
    """Replay-based state recovery depends on this exactly."""
    stream = bernoulli(500, 0.3, seed=71)
    one, two = factory(), factory()
    one.extend(stream)
    for value in stream:
        two.update(value)
    assert one.statistic == pytest.approx(two.statistic)


@pytest.mark.parametrize("factory", DETECTORS)
def test_survives_degenerate_input(factory):
    detector = factory()
    detector.update(0.0)
    assert detector.drift_detected is False
    detector.extend([1.0] * 3)
    assert isinstance(detector.statistic, float)


def test_ddm_reports_a_warning_before_a_drift():
    """The warning level is what a production system would use to start
    buffering data for the retrain."""
    stream = bernoulli(1000, 0.10, seed=81) + bernoulli(1000, 0.45, seed=82)
    detector = DDM()
    saw_warning_first = False
    for value in stream:
        detector.update(value)
        if detector.warning_detected:
            saw_warning_first = True
        if detector.drift_detected:
            break
    assert saw_warning_first


def test_ddm_refuses_to_fire_before_min_instances():
    """Three unlucky errors at the start must not look like a confident
    detection."""
    detector = DDM(min_instances=30)
    detector.extend([1.0] * 10)
    assert detector.drift_detected is False


def test_adwin_rejects_an_invalid_delta():
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="delta"):
            ADWIN(delta=bad)


# --- validation against the reference implementation -------------------------
@pytest.mark.parametrize("error_rate", [0.05, 0.20, 0.40])
def test_matches_river(error_rate):
    """Ours against ``river``'s, on identical seeded streams.

    ``river`` is a test-only dependency: the algorithms are the subject of the
    thesis, so they are implemented here, but agreeing with the reference is
    what makes that defensible rather than merely asserted.
    """
    river_drift = pytest.importorskip("river.drift")

    def alarm_rate(factory, adapt=lambda x: x) -> float:
        fired = 0
        for seed in range(40):
            detector = factory()
            for value in bernoulli(3000, error_rate, seed):
                detector.update(adapt(value))
                if detector.drift_detected:
                    fired += 1
                    break
        return fired / 40

    assert alarm_rate(DDM) == alarm_rate(river_drift.binary.DDM, adapt=bool)
    assert alarm_rate(lambda: ADWIN(delta=0.002)) == alarm_rate(
        lambda: river_drift.ADWIN(delta=0.002)
    )
