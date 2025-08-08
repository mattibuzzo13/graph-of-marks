# igp/nlp.py
from __future__ import annotations

from typing import Iterable, Set, Tuple


def ensure_nltk_corpora(names: Iterable[str] = ("wordnet", "omw-1.4")) -> None:
    """
    Scarica (se assenti) i corpora NLTK necessari. Silenzioso se NLTK non è installato.
    """
    try:
        import nltk

        for n in names:
            try:
                nltk.data.find(f"corpora/{n}")
            except LookupError:
                nltk.download(n, quiet=True)
    except Exception:
        # NLTK non disponibile: semplicemente ignora.
        pass


def ensure_spacy_model(model_name: str = "en_core_web_md") -> bool:
    """
    Verifica che il modello spaCy sia disponibile; prova a scaricarlo se assente.
    Ritorna True se il modello è (o diventa) disponibile.
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
    Estrae (object_terms, relation_terms) dalla domanda.
    - object_terms: token alfabetici non-stopword (fallback minimale)
    - relation_terms: set canonico (above/below/left_of/right_of/on_top_of/…)
    Se spaCy è installato, usa POS tagging per isolare (PROPN/NOUN) in modo più pulito.
    """
    q = (question or "").strip().lower()
    if not q:
        return set(), set()

    # relazioni canoniche + sinonimi base
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

    # prova con spaCy per oggetti (NOUN/PROPN), altrimenti fallback
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
        # Fallback semplice
        tokens = [t for t in q.replace("?", " ").replace(",", " ").split() if t.isalpha()]
        stop = {"the", "a", "an", "is", "are", "on", "in", "of", "to", "and", "or"}
        objs = {t for t in tokens if t not in stop}
        return objs, rel_terms
