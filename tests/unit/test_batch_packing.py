"""Correctness-bit packing.

This is what makes detector state rebuildable *exactly* after a restart, rather
than pickling detector objects whose internals are not a version-stable
contract. 500 messages cost 63 bytes.
"""

import pytest

from streamcls.spark.batch import pack_correct_bits, unpack_correct_bits


@pytest.mark.parametrize(
    "sequence",
    [
        [],
        [1],
        [0],
        [1, 0, 1, 1, 0, 0, 1, 0],
        [1] * 500,
        [0] * 500,
        [i % 3 == 0 for i in range(500)],
    ],
)
def test_round_trip(sequence):
    packed = pack_correct_bits([int(b) for b in sequence])
    assert unpack_correct_bits(packed, len(sequence)) == [int(b) for b in sequence]


def test_a_full_batch_costs_63_bytes():
    """The storage argument for replay-based state recovery."""
    assert len(pack_correct_bits([1] * 500)) == 63


def test_non_multiple_of_eight_does_not_leak_padding():
    sequence = [1, 0, 1]
    assert unpack_correct_bits(pack_correct_bits(sequence), 3) == [1, 0, 1]


def test_empty_and_none_are_handled():
    assert pack_correct_bits([]) == b""
    assert unpack_correct_bits(b"", 10) == []
    assert unpack_correct_bits(None, 10) == []
    assert unpack_correct_bits(pack_correct_bits([1, 1]), 0) == []


def test_accepts_a_memoryview_as_postgres_returns():
    """psycopg2 hands BYTEA back as a memoryview, not bytes."""
    packed = pack_correct_bits([1, 0, 1, 1])
    assert unpack_correct_bits(memoryview(packed), 4) == [1, 0, 1, 1]
