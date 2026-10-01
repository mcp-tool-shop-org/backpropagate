"""Tiny random-weight causal LMs and a word-level tokenizer, built in-process.

No downloads: every model here is constructed from a config with random
weights, and the tokenizer is a ``tokenizers`` WordLevel model wrapped in
``PreTrainedTokenizerFast``. Used by the block-coordinate engine tests.
"""

from __future__ import annotations

import torch

WORDS = [
    "the", "a", "cat", "dog", "sat", "on", "mat", "ran", "to", "park",
    "user", "assistant", "what", "is", "two", "plus", "four", "yes", "no", "and",
]


def tiny_llama(*, tied: bool = False, layers: int = 4, vocab: int = 64, hidden: int = 16,
               seed: int = 0, dtype: torch.dtype = torch.float32):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, tie_word_embeddings=tied,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = LlamaForCausalLM(cfg)
    return model.to(dtype)


def tiny_gpt2(*, layers: int = 3, vocab: int = 64, hidden: int = 16, seed: int = 0,
              dtype: torch.dtype = torch.float32):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=vocab, n_embd=hidden, n_layer=layers, n_head=2, n_positions=128,
                     bos_token_id=1, eos_token_id=2)
    return GPT2LMHeadModel(cfg).to(dtype)


def tiny_tokenizer():
    """A word-level fast tokenizer with pad/bos/eos/unk and a ChatML-ish template."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<s>", "</s>", "<unk>"]
    vocab = {tok: i for i, tok in enumerate(specials + WORDS)}
    tk = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
        unk_token="<unk>",
    )
    tok.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} </s> {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    return tok


def sentences(n: int, seed: int = 0) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        k = int(torch.randint(4, 10, (1,), generator=g))
        idx = torch.randint(0, len(WORDS), (k,), generator=g).tolist()
        out.append(" ".join(WORDS[i] for i in idx))
    return out
