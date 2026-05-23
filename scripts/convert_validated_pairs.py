"""Convert validated_pairs.jsonl to ShareGPT chat format for SFT training.

Each function with doctests becomes a conversation:
- User asks to write a function with tests
- Assistant provides the documented function

This is a reference / one-off script — replace the paths via CLI args before
running. The previous version hardcoded `F:/AI/...` paths from the maintainer's
original rig, which broke on every other machine.

Usage
-----
    python scripts/convert_validated_pairs.py \\
        --input ./output/validated/validated_pairs.jsonl \\
        --output ./output/validated/perfect_pairs_chat.jsonl
"""

import argparse
import json
from pathlib import Path


def format_function_as_chat(item: dict) -> dict:
    """Convert a validated pair to ShareGPT format.

    Creates a natural conversation where:
    - User asks for a function with specific behavior (from doctest examples)
    - Assistant provides the well-documented function
    """
    func_name = item["function_name"]
    func_code = item["function_code"]
    doctest_examples = item.get("doctest_examples", "")

    # Create user prompt from the doctest examples
    # The examples show what the function should do
    if doctest_examples:
        user_prompt = f"""Write a Python function called `{func_name}` that does the following:

```
{doctest_examples}
```

Include a proper docstring with the examples."""
    else:
        user_prompt = f"Write a Python function called `{func_name}` with a proper docstring."

    # Assistant response is the actual function
    assistant_response = f"""Here's the implementation:

```python
{func_code}
```

This function includes doctest examples that demonstrate its behavior."""

    return {
        "conversations": [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": assistant_response}
        ],
        "id": item.get("id", ""),
        "repo": item.get("repo", ""),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert validated_pairs.jsonl into ShareGPT-format chat conversations.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./output/validated/validated_pairs.jsonl"),
        help="JSONL of validated function/doctest pairs (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/validated/perfect_pairs_chat.jsonl"),
        help="Where to write the ShareGPT-format conversations (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    input_path = args.input
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    print(f"Loaded {len(items)} validated pairs")

    # Convert to chat format
    chat_items = []
    for item in items:
        try:
            chat_item = format_function_as_chat(item)
            chat_items.append(chat_item)
        except Exception as e:
            print(f"Error converting {item.get('id', 'unknown')}: {e}")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for item in chat_items:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(chat_items)} chat conversations to {output_path}")

    # Show sample
    print("\n--- Sample conversation ---")
    sample = chat_items[0]
    print(f"User: {sample['conversations'][0]['value'][:200]}...")
    print(f"\nAssistant: {sample['conversations'][1]['value'][:200]}...")


if __name__ == "__main__":
    main()
