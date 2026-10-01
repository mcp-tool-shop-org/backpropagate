"""Regression: ``export_lora`` must not report success for a single-file source.

Passing the adapter's ``adapter_model.safetensors`` FILE (instead of its
directory) used to skip the copy loop entirely, promote an empty ``.partial``
directory to the output path and return an ``ExportResult`` with
``size_mb == 0.0``: a successful-looking export of nothing.

Nothing is mocked: real files on ``tmp_path``.
"""

from __future__ import annotations

import pytest

from backpropagate import export
from backpropagate.exceptions import ExportError


def test_single_file_source_is_rejected_not_exported_as_empty(tmp_path):
    adapter_file = tmp_path / "adapter_model.safetensors"
    adapter_file.write_bytes(b"weights")
    out = tmp_path / "out"
    with pytest.raises(ExportError, match="not a directory") as exc:
        export.export_lora(adapter_file, out, emit_model_card=False)
    assert "adapter directory" in (exc.value.suggestion or "")
    assert not out.exists()
    assert not (tmp_path / "out.partial").exists()


def test_previous_export_is_left_intact_when_the_source_is_a_file(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "adapter_config.json").write_text("{}", encoding="utf-8")
    adapter_file = tmp_path / "adapter_model.safetensors"
    adapter_file.write_bytes(b"weights")
    with pytest.raises(ExportError):
        export.export_lora(adapter_file, out, emit_model_card=False)
    assert (out / "adapter_config.json").exists()  # the earlier good export survives


def test_directory_source_still_works(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "adapter_config.json").write_text("{}", encoding="utf-8")
    (src / "adapter_model.safetensors").write_bytes(b"weights")
    result = export.export_lora(src, tmp_path / "out", emit_model_card=False)
    assert sorted(p.name for p in result.path.iterdir()) == ["adapter_config.json", "adapter_model.safetensors"]
    assert result.size_mb > 0
