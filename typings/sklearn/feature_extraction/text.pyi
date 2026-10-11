from sklearn.base import BaseEstimator

class TfidfVectorizer(BaseEstimator):
    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = ...,
        min_df: int | float = ...,
        stop_words: str | list[str] | None = ...,
    ) -> None: ...

ENGLISH_STOP_WORDS: frozenset[str]
