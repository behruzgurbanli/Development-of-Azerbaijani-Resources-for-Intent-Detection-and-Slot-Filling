"""Local Unicode-aware CRF features derived exclusively from text tokens."""

from collections.abc import Sequence
import unicodedata

from .slots import Token


def word_shape(text: str) -> str:
    """Map uppercase, lowercase, uncased letters, and digits to X/x/L/d."""
    return "".join("X" if c.isupper() else "x" if c.islower() else "L" if c.isalpha()
                   else "d" if c.isdigit() else c for c in text)


def sequence_features(tokens: Sequence[Token], affix_lengths: Sequence[int],
                      context_radius: int) -> list[list[str]]:
    local, affixes = [], []
    for token in tokens:
        word = token.text
        folded = word.casefold()
        category = unicodedata.category
        local.append([
            f"surface={word}", f"casefold={folded}", f"shape={word_shape(word)}",
            f"initial_upper={word[0].isupper()}", f"uppercase={word.isupper()}",
            f"has_digit={any(c.isdigit() for c in word)}", f"all_digits={word.isdigit()}",
            f"punctuation={all(category(c).startswith('P') for c in word)}",
        ])
        features = []
        for n in affix_lengths:
            if n <= len(folded):
                features.extend((f"prefix{n}={folded[:n]}", f"suffix{n}={folded[-n:]}"))
        affixes.append(features)
    output = []
    for index, current in enumerate(local):
        features = ["bias", *current, *affixes[index]]
        for delta in range(-context_radius, context_radius + 1):
            if delta == 0:
                continue
            neighbor = index + delta
            if neighbor < 0:
                features.append(f"{delta}:BOS")
            elif neighbor >= len(local):
                features.append(f"{delta}:EOS")
            else:
                features.extend(f"{delta}:{f}" for f in local[neighbor])
        output.append(features)
    return output
