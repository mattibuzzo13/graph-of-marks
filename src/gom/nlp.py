# igp/nlp.py
"""
Natural Language Processing Utilities

This module provides lightweight NLP utilities for question analysis and term extraction.
Used primarily for question-guided object detection and relationship filtering.

Key Functions:
    - ensure_nltk_corpora: Download NLTK data dependencies
    - ensure_spacy_model: Download and verify spaCy models
    - extract_question_terms: Extract objects and relations from questions

Dependencies:
    - Optional: nltk (for WordNet lemmatization)
    - Optional: spacy (for robust POS tagging and lemmatization)
    - Graceful degradation when dependencies unavailable

Usage:
    >>> from gom.nlp import extract_question_terms
    >>> objs, rels = extract_question_terms("What is on top of the table?")
    >>> print(objs)  # {'table'}
    >>> print(rels)  # {'on_top_of'}
"""
from __future__ import annotations

from typing import Iterable, Set, Tuple


def ensure_nltk_corpora(names: Iterable[str] = ("wordnet", "omw-1.4")) -> None:
    """
    Download required NLTK corpora if not already present.
    
    Ensures necessary NLTK data packages are available for WordNet-based operations.
    Silently succeeds if NLTK is not installed (optional dependency).
    
    Args:
        names: Iterable of corpus names to ensure are downloaded.
              Common corpora:
                - "wordnet": English lexical database for synonyms/hypernyms
                - "omw-1.4": Open Multilingual WordNet data
    
    Behavior:
        - Checks each corpus with nltk.data.find()
        - Downloads missing corpora with nltk.download(quiet=True)
        - No-op if NLTK not installed (graceful degradation)
        - No-op if all corpora already present
    
    Examples:
        >>> ensure_nltk_corpora()  # Download wordnet + omw-1.4
        >>> ensure_nltk_corpora(["punkt", "averaged_perceptron_tagger"])
    
    Notes:
        - Uses quiet=True to suppress download progress
        - Safe to call repeatedly (idempotent)
        - Downloads to default NLTK data directory
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
    Ensure spaCy language model is installed and loadable.
    
    Attempts to load the specified spaCy model. If not found, downloads it
    automatically and then loads it. Used for robust NLP features like POS
    tagging and lemmatization.
    
    Args:
        model_name: spaCy model identifier. Common options:
                   - "en_core_web_sm": Small English model (~13 MB)
                   - "en_core_web_md": Medium English model with vectors (~40 MB)
                   - "en_core_web_lg": Large English model (~560 MB)
                   Default is "md" for balance of size and quality.
    
    Returns:
        True if model successfully loaded (was present or downloaded),
        False if spaCy unavailable or download failed.
    
    Behavior:
        1. Try to load model with spacy.load()
        2. If OSError (model not found), download via spacy.cli.download()
        3. Retry loading after download
        4. Return False on any failure (spaCy not installed, network error, etc.)
    
    Examples:
        >>> if ensure_spacy_model("en_core_web_md"):
        ...     import spacy
        ...     nlp = spacy.load("en_core_web_md")
        ...     doc = nlp("The cat sat on the mat")
        
        >>> # Try small model for faster loading
        >>> ensure_spacy_model("en_core_web_sm")
    
    Notes:
        - Downloads can be large (40-560 MB depending on model)
        - Requires internet connection for download
        - Downloaded models cached in spaCy data directory
        - Safe to call repeatedly (checks before downloading)
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
    Extract object nouns and spatial relations from a natural language question.
    
    Parses a question to identify:
    1. Object terms: Nouns and proper nouns (things to detect)
    2. Relation terms: Spatial/positional relationships (how objects relate)
    
    Used for question-guided visual reasoning to focus detection and relationship
    extraction on relevant objects and predicates.
    
    Args:
        question: Natural language question string (e.g., "What is on the table?")
    
    Returns:
        Tuple of (object_terms, relation_terms) where:
            - object_terms: Set of lowercase lemmatized nouns
            - relation_terms: Set of canonical relation predicates
    
    Algorithm:
        Object Extraction:
            - Preferred: spaCy POS tagging to find NOUN/PROPN, lemmatize, filter stopwords
            - Fallback: Simple tokenization with manual stopword removal
        
        Relation Extraction:
            - Pattern matching against synonym dictionary
            - Maps phrases to canonical predicates:
              * "on top of", "on", "onto" → "on_top_of"
              * "left", "to the left of" → "left_of"
              * "under", "below" → "below"
              * etc.
    
    Examples:
        >>> extract_question_terms("What is on top of the table?")
        ({'table'}, {'on_top_of'})
        
        >>> extract_question_terms("Is the cat to the left of the dog?")
        ({'cat', 'dog'}, {'left_of'})
        
        >>> extract_question_terms("Where is the red car?")
        ({'car'}, set())
        
        >>> extract_question_terms("")
        (set(), set())
    
    Supported Relations:
        - Vertical: above, below/under, on_top_of
        - Horizontal: left_of, right_of
        - Depth: in_front_of, behind
        - Proximity: next_to, touching
    
    Notes:
        - Returns lowercase normalized terms
        - Lemmatizes for consistency (e.g., "cats" → "cat")
        - Removes common stopwords (the, a, is, etc.)
        - Empty question returns (set(), set())
        - Gracefully handles missing spaCy (uses fallback)
    
    Fallback Mode (no spaCy):
        - Splits on whitespace and punctuation
        - Removes short hardcoded stopword list
        - No lemmatization (keeps original forms)
        - Less accurate but functional
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
