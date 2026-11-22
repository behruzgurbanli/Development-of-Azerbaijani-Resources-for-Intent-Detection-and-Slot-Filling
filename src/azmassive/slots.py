"""Exact MASSIVE markup parsing and text-only tokenization for IOB2."""

from dataclasses import dataclass
import re
from collections.abc import Sequence

LABEL = re.compile(r"[a-z][a-z_]*")
MARKUP = re.compile(r"\[([a-z][a-z_]*) : ([^\[\]]+)\]")
TOKENIZERS = {
    "whitespace": re.compile(r"\S+"),
    "word_punctuation": re.compile(r"\w+|[^\w\s]"),
    "keep_joiners": re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]"),
}


@dataclass(frozen=True)
class Span:
    """A typed half-open Python character interval in the original utterance."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def parse_annotation(annotated: str, utt: str) -> tuple[str, tuple[Span, ...]]:
    """Strip markup left to right and require an exact reconstruction of utt."""
    parts = []
    spans = []
    position = offset = 0
    while position < len(annotated):
        char = annotated[position]
        if char == "[":
            match = MARKUP.match(annotated, position)
            if match is None or not match[2].strip() or match[2] != match[2].strip():
                raise ValueError(f"Malformed slot markup at annotation offset {position}")
            label, value = match.groups()
            spans.append(Span(offset, offset + len(value), label))
            position = match.end()
        elif char == "]":
            raise ValueError(f"Unexpected closing bracket at annotation offset {position}")
        else:
            value = char
            position += 1
        parts.append(value)
        offset += len(value)
    reconstructed = "".join(parts)
    if reconstructed != utt:
        raise ValueError("Reconstructed text differs from utt")
    return reconstructed, tuple(spans)


def tokenize(utt: str, method: str = "word_punctuation") -> tuple[Token, ...]:
    """Return source slices with offsets using only the unannotated text."""
    if method not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer: {method}")
    return tuple(Token(m[0], m.start(), m.end()) for m in TOKENIZERS[method].finditer(utt))


class AlignmentError(ValueError):
    def __init__(self, failures: list[dict]):
        super().__init__(f"{len(failures)} unrepresentable slot span(s)")
        self.failures = failures


def spans_to_bio(tokens: Sequence[Token], spans: Sequence[Span]) -> tuple[str, ...]:
    """Require exact token boundaries and report every unrepresentable span."""
    starts = {token.start: i for i, token in enumerate(tokens)}
    ends = {token.end: i for i, token in enumerate(tokens)}
    tags = ["O"] * len(tokens)
    failures = []
    previous_end = 0
    for span in spans:
        left, right = starts.get(span.start), ends.get(span.end)
        reasons = []
        if not LABEL.fullmatch(span.label) or span.end <= span.start:
            reasons.append("invalid_span")
        if span.start < previous_end:
            reasons.append("overlapping_or_unsorted_span")
        if left is None:
            reasons.append("start_not_token_boundary")
        if right is None:
            reasons.append("end_not_token_boundary")
        if left is not None and right is not None and left > right:
            reasons.append("no_covered_tokens")
        previous_end = max(previous_end, span.end)
        if reasons:
            failures.append({
                "start": span.start, "end": span.end, "slot": span.label,
                "reasons": reasons,
                "crossed_tokens": [[t.start, t.end] for t in tokens
                                   if t.start < span.start < t.end or t.start < span.end < t.end],
            })
            continue
        tags[left] = f"B-{span.label}"
        for index in range(left + 1, right + 1):
            tags[index] = f"I-{span.label}"
    if failures:
        raise AlignmentError(failures)
    return tuple(tags)


def bio_to_spans(tokens: Sequence[Token], tags: Sequence[str]) -> tuple[Span, ...]:
    """Decode IOB2 exactly, raising on illegal starts or changes of label."""
    if len(tokens) != len(tags):
        raise ValueError("Token and BIO sequence lengths differ")
    spans = []
    active = None
    for index, (token, tag) in enumerate(zip(tokens, tags)):
        if tag == "O":
            active = None
            continue
        prefix, _, label = tag.partition("-")
        if prefix not in {"B", "I"} or not LABEL.fullmatch(label):
            raise ValueError(f"Invalid BIO tag at token {index}: {tag}")
        if prefix == "B":
            spans.append(Span(token.start, token.end, label))
            active = label
        elif active == label:
            spans[-1] = Span(spans[-1].start, token.end, label)
        else:
            raise ValueError(f"Invalid IOB2 transition at token {index}: {tag}")
    return tuple(spans)
