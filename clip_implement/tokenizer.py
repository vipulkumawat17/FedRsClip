from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch


@dataclass(frozen=True)
class TokenizedText:
    input_ids: torch.LongTensor


class ByteTokenizer:
    """Small lower-cased byte tokenizer with CLIP-style SOS/EOS bracketing.

    The CLIP paper used lower-cased BPE with a 49,152 token vocabulary and a
    context length of 76. This byte-level tokenizer keeps the same sequence
    interface without requiring external vocab files.
    """

    pad_token_id = 0
    sos_token_id = 1
    eos_token_id = 2
    byte_offset = 3

    def __init__(self, context_length: int = 76) -> None:
        self.context_length = context_length
        self.vocab_size = 259

    def encode(self, text: str) -> List[int]:
        text = text.lower().strip()
        byte_ids = [b + self.byte_offset for b in text.encode("utf-8", errors="replace")]
        ids = [self.sos_token_id, *byte_ids, self.eos_token_id]
        return ids[: self.context_length]

    def __call__(self, texts: str | Iterable[str]) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        output = []
        for text in texts:
            ids = self.encode(text)
            padded = ids + [self.pad_token_id] * (self.context_length - len(ids))
            if self.eos_token_id not in padded:
                padded[-1] = self.eos_token_id
            output.append(padded)
        return torch.tensor(output, dtype=torch.long)
