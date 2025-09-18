"""
Query Transformations: tecniche per trasformare una domanda in varianti utili al retrieval
Ispirato all'implementazione generale: NirDiamant/RAG_Techniques
"""
from dataclasses import dataclass
from typing import List
import logging
import re

from config.config import QueryTransformationsConfig


class QueryTransformer:
    """
    Applica trasformazioni alla query per migliorarne il recupero:
    - Decompose: spezza query complesse in sotto-domande
    - Rewrite: riformula la query
    - Expand: espande con parole chiave correlate
    """

    def __init__(self, config: QueryTransformationsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def transform(self, query: str) -> List[str]:
        variants: List[str] = []

        base = query.strip()
        if not base:
            return []

        # Mantieni l'originale sempre come prima variante
        variants.append(base)

        if self.config.enable_decompose:
            variants.extend(self._decompose(base))

        if self.config.enable_rewrite:
            variants.extend(self._rewrite(base))

        if self.config.enable_expand:
            variants.extend(self._expand(base))

        # Normalizza: rimuovi duplicati, vuoti, limita a max_transformations
        cleaned = []
        seen = set()
        for v in variants:
            v2 = re.sub(r"\s+", " ", v).strip()
            if v2 and v2.lower() not in seen:
                seen.add(v2.lower())
                cleaned.append(v2)

        if self.config.max_transformations > 0:
            cleaned = cleaned[: self.config.max_transformations]

        self.logger.debug(f"Generate {len(cleaned)} varianti di query")
        return cleaned

    def _decompose(self, query: str) -> List[str]:
        # Heuristica semplice: spezza su connettivi comuni
        connectors = [
            " e ", " ed ", " oppure ", " o ", " then ", " and ", " or ", "; ", ". ", ", "
        ]
        parts = [query]
        for c in connectors:
            if c in query:
                parts = [p.strip() for p in query.split(c) if p.strip()]
                break
        return [p for p in parts if len(p.split()) > 2 and p.lower() != query.lower()]

    def _rewrite(self, query: str) -> List[str]:
        # Riformulazioni semplici via pattern
        q = query.rstrip(" ?!.")
        rewrites = [
            f"Spiega {q}",
            f"Descrivi {q}",
            f"Dettagli su {q}",
            f"Come funziona {q}?",
        ]
        return rewrites

    def _expand(self, query: str) -> List[str]:
        # Estrai potenziali keyword e crea query espanse
        tokens = re.findall(r"\b\w+\b", query.lower())
        keywords = [t for t in tokens if len(t) > 3]
        expansions: List[str] = []
        if keywords:
            expansions.append(f"{query} dettagli configurazione")
            expansions.append(f"{query} problemi comuni")
            expansions.append(f"{query} esempio passo passo")
        return expansions


def create_query_transformer(config: QueryTransformationsConfig) -> QueryTransformer:
    return QueryTransformer(config)
