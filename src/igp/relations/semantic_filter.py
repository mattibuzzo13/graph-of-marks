"""Semantic filtering helpers for relations.

Provides a best-effort, optional WordNet-backed validator to detect animate
labels and wearable objects. Gracefully falls back to small curated lists
when NLTK/WordNet is unavailable or label lookup fails.
"""
from typing import List, Sequence

try:
    # NLTK's WordNet provides synsets and hypernym paths which we can use
    # to infer coarse-grained categories like 'person' or 'clothing'. This
    # import is optional; fallback lists are used when unavailable.
    from nltk.corpus import wordnet as wn  # type: ignore
    _WN_AVAILABLE = True
except Exception:
    wn = None  # type: ignore
    _WN_AVAILABLE = False


# Small curated fallbacks
_ANIMATE_TOKENS = {
    "person",
    "people",
    "man",
    "woman",
    "child",
    "boy",
    "girl",
    "human",
    "dog",
    "cat",
    "person",
}

_WEARABLE_TOKENS = {
    "hat",
    "cap",
    "glasses",
    "sunglasses",
    "shirt",
    "jacket",
    "coat",
    "shoe",
    "shoes",
    "pants",
    "skirt",
    "dress",
    "tie",
    "scarf",
    "watch",
    "belt",
}


def _label_synsets(label: str):
    if not _WN_AVAILABLE:
        return []
    try:
        return wn.synsets(label)
    except Exception:
        return []


def is_animate(label: str) -> bool:
    """Return True if label likely denotes an animate entity.

    Attempts WordNet-based checks first, then falls back to token matching.
    This is conservative: prefer false-negatives over false-positives.
    """
    if not label:
        return False
    norm = label.lower().strip()
    # quick token check
    if any(tok == norm or tok in norm for tok in _ANIMATE_TOKENS):
        return True

    # WordNet inference: check if any synset has a hypernym path including
    # 'person' or 'animal' or 'organism' nodes.
    syns = _label_synsets(norm)
    for s in syns:
        try:
            for path in s.hypernym_paths():
                for h in path:
                    name = h.name().split(".")[0].lower()
                    if name in ("person", "animal", "organism", "living_thing", "human"):
                        return True
        except Exception:
            continue

    return False


def is_wearable(label: str) -> bool:
    """Return True if label likely denotes a wearable/clothing item.

    Uses WordNet where available, otherwise falls back to curated tokens.
    """
    if not label:
        return False
    norm = label.lower().strip()
    if any(tok == norm or tok in norm for tok in _WEARABLE_TOKENS):
        return True

    syns = _label_synsets(norm)
    for s in syns:
        try:
            # check if clothing/clothes/clothing.n.01 is in hypernyms
            for path in s.hypernym_paths():
                for h in path:
                    name = h.name().split(".")[0].lower()
                    if name in ("clothing", "garment", "apparel", "accessory"):
                        return True
        except Exception:
            continue

    return False


def filter_impossible_relations(relationships: List[dict], labels: Sequence[str]) -> List[dict]:
    """Filter relations conservatively using semantic checks.

    - Relations that require animate subjects (e.g., 'wearing', 'holding') are
      dropped when the subject label is unlikely animate.
    - 'wearing' relations require the object to be wearable.
    - This function is intentionally conservative to avoid removing plausible
      relations.
    """
    if not relationships:
        return relationships

    require_animate_subj = {"wearing", "holding", "riding", "sitting_on", "carrying"}

    out = []
    for r in relationships:
        rel = str(r.get("relation", "")).lower()
        s_idx = int(r.get("src_idx", -1))
        t_idx = int(r.get("tgt_idx", -1))

        subj_label = labels[s_idx] if 0 <= s_idx < len(labels) else ""
        obj_label = labels[t_idx] if 0 <= t_idx < len(labels) else ""

        # animate requirement
        if rel in require_animate_subj:
            if not is_animate(subj_label):
                # drop implausible relation
                continue

        if rel == "wearing":
            if not is_wearable(obj_label):
                continue

        out.append(r)

    return out
