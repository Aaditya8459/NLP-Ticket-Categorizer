"""Text preprocessing pipeline: clean, normalize, tokenize, stem."""

import re
import string
import logging
from typing import List, Set

from src.config import Config

logger = logging.getLogger(__name__)

# Comprehensive English stopword list (no external deps)
BUILTIN_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "up", "about", "into", "through", "during", "before", "after", "above", "below",
    "between", "among", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used", "i", "me", "my", "myself", "we", "our", "ours",
    "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "so",
    "than", "too", "very", "just", "now", "then", "here", "there", "when", "where", "why", "how",
    "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "also", "get", "got", "go", "going", "come", "came", "say", "said", "know",
    "think", "see", "look", "way", "time", "day", "week", "month", "year", "hi", "hello", "dear",
    "team", "please", "thanks", "thank", "regards", "best", "kindly", "like", "want", "help", "us",
    "let", "make", "take", "give", "tell", "ask", "asked", "asking", "told", "telling", "saying",
    "coming", "getting", "making", "taking", "giving", "use", "using", "used", "work", "worked",
    "working", "call", "called", "calling", "try", "tried", "trying", "find", "found", "finding",
    "put", "puts", "putting", "set", "sets", "setting", "keep", "kept", "keeping", "seem", "seemed",
    "seeming", "feel", "felt", "feeling", "become", "became", "becoming", "leave", "left", "leaving",
    "good", "new", "first", "last", "long", "great", "little", "own", "old", "right", "big", "high",
    "different", "small", "large", "next", "early", "young", "important", "few", "public", "bad",
    "same", "able"
}


class TextPreprocessor:
    """
    Production-grade text preprocessor.

    Handles cleaning, normalization, stopword removal, and stemming
    without external dependencies (unless Porter stemmer is requested).
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.stopwords = self._load_stopwords()
        self.stem_func = self._get_stemmer()
        logger.info("TextPreprocessor initialized (stemmer=%s)", config.prep_stemmer)

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords from builtin list or external file."""
        if self.cfg.prep_stopwords_src == "builtin":
            return BUILTIN_STOPWORDS
        # External file path
        try:
            with open(self.cfg.prep_stopwords_src, "r", encoding="utf-8") as f:
                return set(line.strip().lower() for line in f if line.strip())
        except FileNotFoundError:
            logger.warning("Stopwords file not found, using builtin. Path: %s", 
                          self.cfg.prep_stopwords_src)
            return BUILTIN_STOPWORDS

    def _get_stemmer(self):
        """Return the appropriate stemming function."""
        stemmer_type = self.cfg.prep_stemmer.lower()

        if stemmer_type == "none":
            return lambda w: w

        if stemmer_type == "porter":
            try:
                from nltk.stem import PorterStemmer
                ps = PorterStemmer()
                logger.info("Using NLTK PorterStemmer")
                return ps.stem
            except ImportError:
                logger.warning("nltk not installed, falling back to simple stemmer")
                return self._simple_stem

        return self._simple_stem

    @staticmethod
    def _simple_stem(word: str) -> str:
        """
        Lightweight suffix-stripping stemmer (Porter-style rules).
        No dependencies, fast, good enough for most use cases.
        """
        suffixes = [
            ("ies", "y"), ("ied", "y"), ("ying", "y"),
            ("s", ""), ("es", ""), ("ed", ""), ("ing", ""),
            ("er", ""), ("est", ""), ("ly", ""), ("tion", "t"),
            ("ness", ""), ("ment", ""), ("able", ""), ("ible", ""),
            ("ful", ""), ("less", ""), ("ize", ""), ("ise", ""),
            ("ity", ""), ("ties", "ty"), ("al", ""), ("ism", ""),
            ("ist", ""), ("ous", ""), ("ive", ""), ("ize", ""),
            ("ised", "ize"), ("ized", "ize")
        ]
        for suf, repl in suffixes:
            if word.endswith(suf) and len(word) > len(suf) + 2:
                return word[:-len(suf)] + repl
        return word

    def clean(self, text: str) -> str:
        """
        Full preprocessing pipeline on raw text.

        Steps:
            1. Lowercase
            2. Remove URLs
            3. Remove email addresses
            4. Remove phone numbers
            5. Remove numeric tokens (including currency)
            6. Remove punctuation
            7. Tokenize, filter stopwords & short tokens
            8. Stem

        Args:
            text: Raw input string.

        Returns:
            Cleaned, space-joined string ready for vectorization.
        """
        if not isinstance(text, str):
            text = str(text)

        # 1. Lowercase
        if self.cfg.prep_lowercase:
            text = text.lower()

        # 2. Remove URLs
        if self.cfg.prep_remove_urls:
            text = re.sub(r"http[s]?://\S+", "", text)

        # 3. Remove emails
        if self.cfg.prep_remove_emails:
            text = re.sub(r"\S+@\S+", "", text)

        # 4. Remove phone numbers (various formats)
        if self.cfg.prep_remove_phones:
            text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "", text)

        # 5. Remove numbers (including $, €, £ prefixed)
        if self.cfg.prep_remove_numbers:
            text = re.sub(r"[\$€£]?\d+[\.,]?\d*", "", text)

        # 6. Remove punctuation
        if self.cfg.prep_remove_punct:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 7. Tokenize + filter
        tokens = text.split()
        min_len = self.cfg.prep_min_token_len
        tokens = [t for t in tokens if t not in self.stopwords and len(t) >= min_len]

        # 8. Stem
        tokens = [self.stem_func(t) for t in tokens]

        return " ".join(tokens)

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts."""
        return [self.clean(t) for t in texts]
