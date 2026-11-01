import pytest

from astrai.preprocessing import generate_position_ids


@pytest.mark.parametrize(
    "sequences, mode, expected",
    [
        ([[7, 8], [], [9, 10, 11]], "none", []),
        ([[7, 8], [], [9, 10, 11]], "doc_reset", [0, 1, 0, 1, 2]),
        ([[7, 8], [], [9, 10, 11]], "continuous", [0, 1, 2, 3, 4]),
        ([], "none", []),
        ([], "doc_reset", []),
        ([], "continuous", []),
        ([[], []], "none", []),
        ([[], []], "doc_reset", []),
        ([[], []], "continuous", []),
    ],
)
def test_position_modes_preserve_document_boundaries(sequences, mode, expected):
    original = [list(sequence) for sequence in sequences]
    assert generate_position_ids(sequences, mode) == expected
    assert sequences == original


def test_position_ids_reject_unknown_mode():
    with pytest.raises(ValueError, match="Unknown position_ids_mode"):
        generate_position_ids([], "unknown")
