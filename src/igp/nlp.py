# igp/nlp.py
from __future__ import annotations

from typing import Iterable, Set, Tuple


def ensure_nltk_corpora(names: Iterable[str] = ("wordnet", "omw-1.4")) -> None:
    """
    Download required NLTK corpora if missing. Silent no-op when NLTK is not installed.
    """
    try:
        import nltk

        for n in names:
            try:
                nltk.data.find(f"corpora/{n}")
            except LookupError:
                nltk.download(n, quiet=True)
    except Exception:
        # NLTK unavailable: ignore.
        pass


def ensure_spacy_model(model_name: str = "en_core_web_md") -> bool:
    """
    Ensure a spaCy model is available; attempt on-demand download if absent.
    Returns True on success (loaded or downloaded), False otherwise.
    """
    try:
        import spacy

        try:
            spacy.load(model_name)
            return True
        except OSError:
            from spacy.cli import download

            download(model_name, quiet=True)
            spacy.load(model_name)
            return True
    except Exception:
        return False


def extract_question_terms(question: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract (object_terms, relation_terms) from a question.
    - object_terms: alphabetic tokens excluding simple stopwords (fallback)
      or NOUN/PROPN lemmas via spaCy if available.
    - relation_terms: canonical labels inferred from simple synonyms
      (above/below/left_of/right_of/on_top_of/...).
    """
    q = (question or "").strip().lower()
    if not q:
        return set(), set()

    # Canonical relations + basic synonyms
    rel_map = {
        "above": {"above"},
        "below": {"below", "under"},
        "left_of": {"left", "to the left of"},
        "right_of": {"right", "to the right of"},
        "on_top_of": {"on top of", "on", "onto", "resting on", "sitting on"},
        "in_front_of": {"in front of"},
        "behind": {"behind"},
        "next_to": {"next to", "beside"},
        "touching": {"touching"},
    }
    rel_terms = {canon for canon, vs in rel_map.items() if any(v in q for v in vs)}

    # Prefer spaCy (NOUN/PROPN lemmas); fallback to a minimal heuristic
    try:
        import spacy

        nlp = spacy.load("en_core_web_md")
        doc = nlp(q)
        objs = {
            t.lemma_.lower()
            for t in doc
            if t.pos_ in {"NOUN", "PROPN"} and not t.is_stop and t.is_alpha
        }
        return objs, rel_terms
    except Exception:
        # Simple fallback
        tokens = [t for t in q.replace("?", " ").replace(",", " ").split() if t.isalpha()]
        stop = {"the", "a", "an", "is", "are", "on", "in", "of", "to", "and", "or"}
        objs = {t for t in tokens if t not in stop}
        return objs, rel_terms
