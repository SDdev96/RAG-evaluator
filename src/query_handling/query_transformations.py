"""
Query Transformations: tecniche per trasformare una domanda in varianti utili al retrieval
Ispirato all'implementazione generale: NirDiamant/RAG_Techniques
"""
from dataclasses import dataclass
from typing import List, Optional
import logging
import re
import os

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config.config import QueryTransformationsConfig, GenerationConfig


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
        self._gemini_model = None  # lazy init

    def transform(self, query: str) -> tuple[List[str], dict]:
        """Trasforma la query in diverse varianti.
        
        Returns:
            tuple: (lista_di_varianti, llm_datas)
            dove llm_datas è un dizionario che contiene:
            - prompr_used: str
            - transformations: list<dict>
            - transformations[i]: {type: str, query: str}

            Return example:
            (['query1', 'query2'], 
             {'prompt_used': 'prompt', 'transformations': [{'type': 'original', 'query': 'original_query'}, {'type': 'decomposed', 'query': 'transformed_query'}]})
        """
        variants: List[str] = []
        llm_datas = {
            'prompt_used': '',
            'transformations': [],
            'input_tokens': 0,
            'output_tokens': 0
        }

        base = query.strip()
        if not base:
            return [], llm_datas

        # Mantieni l'originale sempre come prima variante
        # variants.append(base)
        llm_datas['transformations'].append({
            'type': 'original',
            'query': base
        })

        if self.config.enable_decompose:
            decomposed = self._decompose(base)
            self.logger.info(f"Decompose: generate {len(decomposed)} varianti -> {decomposed}")
            print(f"[QueryTransform] Decompose ({len(decomposed)}): {decomposed}")
            variants.extend(decomposed)
            llm_datas['transformations'].extend([
                {'type': 'decomposed', 'query': d} for d in decomposed
            ])

        if self.config.enable_rewrite:
            rewritten, llm_response, prompt_used, token_info = self._rewrite(base)
            self.logger.info(f"Rewrite: generate {len(rewritten)} varianti -> {rewritten}")
            print(f"[QueryTransform] Rewrite ({len(rewritten)}): {rewritten}")
            variants.extend(rewritten)
            llm_datas['prompt_used'] = prompt_used
            llm_datas['input_tokens'] = token_info.get('input_tokens', 0)
            llm_datas['output_tokens'] = token_info.get('output_tokens', 0)
            llm_datas['transformations'].extend([
                {**{'type': 'rewritten', 'query': r}, **token_info} for r in rewritten
            ])

        if self.config.enable_expand:
            expanded = self._expand(base)
            self.logger.info(f"Expand: generate {len(expanded)} varianti -> {expanded}")
            print(f"[QueryTransform] Expand ({len(expanded)}): {expanded}")
            variants.extend(expanded)
            llm_datas['transformations'].extend([
                {'type': 'expanded', 'query': e} for e in expanded
            ])

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

        self.logger.info(f"Generate {len(cleaned)} varianti di query")
        return cleaned, llm_datas

    def _decompose(self, query: str) -> List[str]:
        """Decomposizione tramite Gemini con fallback euristico."""
        try:
            n = max(1, int(self.config.max_transformations))
            lang = getattr(self.config, "language", "it")
            prompt = (
                f"Sei un assistente AI incaricato di scomporre query complesse in sottoquery più semplici per un sistema RAG.\n"
                f"Data la query originale, scomponila in {n} sottoquery più semplici che, se risposte insieme, fornirebbero una risposta completa alla query originale.\n\n"
                f"Query originale: {query}\n\n"
                f"Rispondi in {lang}.\n"
                f"Rispondi solo con un elenco numerato di {n} sottoquery, una per riga, senza testo aggiuntivo."
            )
            response_text = self._gemini_generate(prompt)
            if response_text:
                subs = self._parse_numbered_list(response_text)
                return [s for s in subs if s and s.lower() != query.lower()]
        except Exception as e:
            self.logger.warning(f"Fallback decomposizione euristica per errore Gemini: {e}")

        # Fallback euristico: spezza su connettivi comuni
        connectors = [" e ", " ed ", " oppure ", " o ", " then ", " and ", " or ", "; ", ". ", ", "]
        parts = [query]
        for c in connectors:
            if c in query:
                parts = [p.strip() for p in query.split(c) if p.strip()]
                break
        return [p for p in parts if len(p.split()) > 2 and p.lower() != query.lower()]

    def _rewrite(self, query: str) -> tuple[List[str], str, str, dict]:
        """Riformulazione tramite Gemini.
        
        Returns:
            tuple: (varianti_query, llm_response, prompt_used, token_info)
            dove token_info è un dizionario con input_tokens e output_tokens
        """
        try:
            lang = getattr(self.config, "language", "it")
            prompt= f"""
            Sei un assistente specializzato nella riformulazione delle query per un sistema di Retrieval-Augmented Generation (RAG).
            Il tuo compito è:
                1. Analizzare la query originale.
                2. Individuare le parole chiavi principali e i concetti rilevanti.
                3. Riscrivere la query in forma chiara, specifica e contestualizzata e ampliandone il contesto senza cambiare il significato.
                4. Le parole tra apici ' ' o virgolette " " e acronimi devono essere riutilizzate nella riformulazione della domanda.
                5. Eliminare eventuali ambiguità, ridondanze e termini superflui.
                6. Se utile, esplicitare i sinonimi, varianti lessicali o acronimi per migliorarne il recupero.
                7. Ti verrà passata solo la query originale, individuata da "Query originale: "
                8. La tua risposta dovrà avere solo la domanda riformulata, senza aggiungere altro.

            Query originale: {query}
            """
            # prompt2 = (
            #     "Sei un assistente AI incaricato di riformulare le query degli utenti per migliorarne il recupero in un sistema RAG.\n"
            #     "Data la 'query originale', riscrivila, sempre sotto forma di domanda, per renderla più specifica, dettagliata e in grado di recuperare informazioni rilevanti.\n\n"
            #     f"Rispondi in {lang}.\n"
            #     f"Scrivi solo la domanda senza aggiungere altro\n\n"
            #     f"Query originale: {query}\n"
            # )
            response_text, prompt_used, token_info = self._gemini_generate(prompt)
            if response_text:
                # Prendi la prima riga non vuota come riformulazione
                lines = [l.strip("- *# ") for l in response_text.splitlines() if l.strip()]
                if lines:
                    return [lines[0]], response_text, prompt_used, token_info
        except Exception as e:
            self.logger.warning(f"Fallback rewrite euristico per errore Gemini: {e}")

        # Fallback semplice
        q = query.rstrip(" ?!.")
        fallback_queries = [f"Spiega {q}", f"Descrivi {q}", f"Dettagli su {q}", f"Come funziona {q}?"]
        return fallback_queries, "Fallback to simple rewrite", "", {'input_tokens': 0, 'output_tokens': 0}

    def _expand(self, query: str) -> List[str]:
        """Step-back tramite Gemini (query più generali)."""
        try:
            lang = getattr(self.config, "language", "it")
            prompt = (
                "Sei un assistente AI incaricato di generare query più ampie e generali per migliorare il recupero del contesto in un sistema RAG.\n"
                "Data la query originale, genera una query step-back più generale che possa aiutare a recuperare informazioni di base rilevanti.\n\n"
                f"Query originale: {query}\n\n"
                f"Rispondi in {lang}.\n"
                "Query step-back:"
            )
            response_text = self._gemini_generate(prompt)
            if response_text:
                lines = [l.strip("- *# ") for l in response_text.splitlines() if l.strip()]
                if lines:
                    return [lines[0]]
        except Exception as e:
            self.logger.warning(f"Fallback expand euristico per errore Gemini: {e}")

        # Fallback: espansioni keyword-based
        tokens = re.findall(r"\b\w+\b", query.lower())
        keywords = [t for t in tokens if len(t) > 3]
        expansions: List[str] = []
        if keywords:
            expansions.append(f"{query} dettagli configurazione")
            expansions.append(f"{query} problemi comuni")
            expansions.append(f"{query} esempio passo passo")
        return expansions

    def _gemini_generate(self, prompt: str) -> tuple[Optional[str], str, dict]:
        """Esegue una chiamata a Gemini e restituisce il testo, il prompt e i token usati.
        
        Returns:
            tuple: (testo_risposta, prompt_usato, token_info)
            dove token_info è un dizionario con:
            - input_tokens: numero di token in input
            - output_tokens: numero di token in output
        """
        model = self._get_gemini_model()
        token_info = {'input_tokens': 0, 'output_tokens': 0}
        try:
            resp = model.generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                # Stima dei token (approssimativa, in quanto l'API Gemini non fornisce direttamente il conteggio)
                token_info['input_tokens'] = len(prompt.split())  # Approssimazione
                token_info['output_tokens'] = len(resp.text.strip().split())  # Approssimazione
                return resp.text.strip(), prompt, token_info
        except Exception as e:
            self.logger.error(f"Errore chiamata Gemini: {e}")
        return None, prompt, token_info

    def _get_gemini_model(self):
        """Inizializza pigramente il modello Gemini usando GOOGLE_API_KEY."""
        if self._gemini_model is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("API key di Google non fornita (env GOOGLE_API_KEY)")
            genai.configure(api_key=api_key)
            # Usa la stessa configurazione del generatore per coerenza
            gcfg = GenerationConfig()
            self._gemini_model = genai.GenerativeModel(
                model_name=gcfg.model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=gcfg.temperature,
                    max_output_tokens=min(300, gcfg.max_tokens),
                    top_p=gcfg.top_p,
                    top_k=gcfg.top_k,
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                },
            )
            self.logger.info(f"Modello Gemini per QueryTransformations inizializzato ({gcfg.model_name})")
        return self._gemini_model

    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parsa liste numerate in sottodomande."""
        items: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            # match pattern tipo '1. domanda', '- domanda', '* domanda'
            m = re.match(r"^(?:\d+\.|[-*])\s+(.*)$", line)
            if m:
                items.append(m.group(1).strip())
        # Se non ha trovato pattern numerati, ritorna righe non vuote
        if not items:
            items = [l.strip("- *# ") for l in text.splitlines() if l.strip()]
        return items


def create_query_transformer(config: QueryTransformationsConfig) -> QueryTransformer:
    return QueryTransformer(config)
