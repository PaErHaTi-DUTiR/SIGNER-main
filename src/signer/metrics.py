"""Exact original-character boundary-and-type scoring; no token-projected gold."""
from collections import defaultdict


def transcript(row):
    text = row["text"]
    return text if isinstance(text, str) else text["normalized_transcript"]


def span_set(row):
    text = transcript(row)
    result = set()
    for e in row.get("entities", []):
        a, b, label = e["start"], e["end"], e["label"]
        if not 0 <= a < b <= len(text) or ("text" in e and text[a:b] != e["text"]):
            raise ValueError("Entity coordinates/surface do not match original transcript")
        result.add((a, b, label))
    return result


def counts_to_metrics(counts):
    p, g, m = counts
    return dict(predicted=p, reference=g, matched=m, precision=m/p if p else 0.,
                recall=m/g if g else 0., micro_f1=2*m/(p+g) if p+g else 0.)


def score(predictions, references):
    p = {r["id"]: r for r in predictions}; g = {r["id"]: r for r in references}
    if len(p) != len(predictions) or len(g) != len(references) or set(p) != set(g):
        raise ValueError("Duplicate/mismatched sample IDs")
    languages, labels = defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0])
    for ident, row in g.items():
        other = p[ident]
        if row["language"] != other["language"] or transcript(row) != transcript(other):
            raise ValueError("Text/language identity mismatch")
        pp, gg = span_set(other), span_set(row)
        key = row["language"]
        languages[key] = [a+b for a, b in zip(languages[key], (len(pp), len(gg), len(pp & gg)))]
        for label in {"PER", "LOC", "ORG"} | {s[2] for s in pp | gg}:
            a, b = {s for s in pp if s[2] == label}, {s for s in gg if s[2] == label}
            labels[label] = [x+y for x, y in zip(labels[label], (len(a), len(b), len(a & b)))]
    per_language = {k: counts_to_metrics(v) for k, v in sorted(languages.items())}
    total = [sum(v[i] for v in languages.values()) for i in range(3)]
    return dict(overall=counts_to_metrics(total), per_language=per_language,
                language_macro_f1=sum(v["micro_f1"] for v in per_language.values()) / len(per_language) if per_language else 0.,
                per_entity={k: counts_to_metrics(v) for k, v in sorted(labels.items())},
                scoring="exact_original_character_span_and_type", samples=len(g))
