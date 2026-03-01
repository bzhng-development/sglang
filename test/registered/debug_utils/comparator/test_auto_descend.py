import torch
from pathlib import Path

from sglang.srt.debug_utils.comparator.entrypoint import _auto_descend_dir


def _make_pt(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor([1.0]), directory / "dummy.pt")


def test_no_descend_when_pt_at_root(tmp_path: Path) -> None:
    _make_pt(tmp_path)
    _make_pt(tmp_path / "child_a")

    result: Path = _auto_descend_dir(tmp_path, label="test")
    assert result == tmp_path


def test_descend_into_single_child(tmp_path: Path) -> None:
    child: Path = tmp_path / "engine_0"
    _make_pt(child)

    result: Path = _auto_descend_dir(tmp_path, label="test")
    assert result == child


def test_no_descend_with_multiple_children(tmp_path: Path) -> None:
    _make_pt(tmp_path / "engine_0")
    _make_pt(tmp_path / "engine_1")

    result: Path = _auto_descend_dir(tmp_path, label="test")
    assert result == tmp_path


def test_no_descend_when_no_children_have_pt(tmp_path: Path) -> None:
    (tmp_path / "empty_child").mkdir()

    result: Path = _auto_descend_dir(tmp_path, label="test")
    assert result == tmp_path
