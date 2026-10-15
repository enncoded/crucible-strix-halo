"""abab.py depth reads the mtp_tune depth prompts, which hold comma-separated ids (night 3, Oct 10: all depth steps
failed with ValueError because the file was split on white space only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.mtp_snapshot.abab import read_ids  # noqa: E402


def test_comma_separated_and_spaced_ids_read_the_same(tmp_path):
    comma, spaced = tmp_path / "a.txt", tmp_path / "b.txt"
    comma.write_text("248045,846,198\n1022,34313\n")
    spaced.write_text("248045 846 198\n1022 34313\n")
    assert read_ids(comma) == read_ids(spaced) == [248045, 846, 198, 1022, 34313]


def test_depth_takes_the_stream_count(monkeypatch):
    # one stream fits 196K on the daily budget (two streams at ctx 200704 exceeded it, night3d2 Oct 10).
    from bench.mtp_snapshot import abab
    seen = []
    monkeypatch.setattr(abab, "depth", lambda args: seen.append(args.streams) or 0)
    abab.main(["depth", "e", "c", "ids", "out", "--streams", "1"])
    abab.main(["depth", "e", "c", "ids", "out"])
    assert seen == [1, 2]


def test_a_row_keeps_the_cache_bytes():
    from bench.mtp_snapshot import abab
    result = dict(done=dict(decode_seconds=1.0, cache_bytes=123, cached_tokens=8), output_ids=[1, 2, 3], ttft=0.5)
    assert abab.row("x", [1], 0, result, {})["cache_bytes"] == 123


def test_depth_refuses_a_stream_count_out_of_range():
    import pytest
    from bench.mtp_snapshot import abab
    for value in ("0", "9", "x"):
        with pytest.raises(SystemExit):
            abab.main(["depth", "e", "c", "ids", "out", "--streams", value])
