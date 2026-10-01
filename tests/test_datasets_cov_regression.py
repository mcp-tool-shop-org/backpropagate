"""Regression: an empty CSV/parquet cell must not train the model on ``'None'``.

DATA-B-002 coerces empty cells to ``None`` and promises "the converters
already skip" them. They did not: ``dict.get(key, "")`` only defaults for an
ABSENT key, so a present-but-``None`` ``output`` / ``value`` / ``chosen`` was
interpolated into the ChatML text as the four characters ``None``.

Nothing is mocked: real CSV on ``tmp_path`` through ``DatasetLoader``.
"""

from __future__ import annotations

import logging

from backpropagate.datasets import DatasetLoader, FormatConverter


def test_alpaca_none_output_renders_empty_not_literal_none():
    text = FormatConverter.alpaca_to_chatml({"instruction": "hi", "input": None, "output": None})
    assert text == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<|im_end|>"
    assert "None" not in text


def test_alpaca_none_instruction_renders_empty():
    text = FormatConverter.alpaca_to_chatml({"instruction": None, "output": "bye"})
    assert text == "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\nbye<|im_end|>"


def test_sharegpt_none_value_renders_empty():
    text = FormatConverter.sharegpt_to_chatml(
        {"conversations": [{"from": "human", "value": None}, {"from": "gpt", "value": "ok"}]}
    )
    assert text == "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\nok<|im_end|>"


def test_preference_none_chosen_renders_empty():
    text = FormatConverter.preference_to_chatml({"prompt": "q", "chosen": None, "rejected": "r"})
    assert text == "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<|im_end|>"


def test_zero_and_false_values_are_not_swallowed():
    """Only ``None`` is mapped to empty; legitimate falsy values survive."""
    text = FormatConverter.alpaca_to_chatml({"instruction": "n?", "output": 0})
    assert text.endswith("<|im_start|>assistant\n0<|im_end|>")


def test_csv_empty_cell_flows_through_loader_without_none_text(tmp_path, caplog):
    csv = tmp_path / "d.csv"
    csv.write_text("instruction,output\nhello,world\nfoo,\n", encoding="utf-8")
    loader = DatasetLoader(csv)
    with caplog.at_level(logging.WARNING, logger="backpropagate.datasets"):
        chatml = loader.to_chatml()
    assert chatml[1]["text"] == "<|im_start|>user\nfoo<|im_end|>\n<|im_start|>assistant\n<|im_end|>"
    # the blank answer is still surfaced to the operator as an empty turn
    assert any("empty assistant turn" in r.getMessage() for r in caplog.records)
