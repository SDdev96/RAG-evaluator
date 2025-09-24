"""
Langfuse setup utilities using CallbackHandler, based on the provided snippet.

Usage:
    from src.telemetry.langfuse_setup import init_langfuse
    langfuse, langfuse_handler = init_langfuse()
    # Then pass `langfuse_handler` in LangChain calls, e.g.:
    # chain.invoke({"input": user_input}, config={"callbacks": [langfuse_handler]})
"""
# from __future__ import annotations

import os
from typing import Optional, Tuple
from dotenv import load_dotenv

# External SDK
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

# Load .env to allow LANGFUSE_* variables
load_dotenv()

# Environment-based configuration (preferred)
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
LF_HOST = "https://cloud.langfuse.com"


def init_langfuse() -> Tuple[Optional[Langfuse], Optional[CallbackHandler]]:
    """Initialize Langfuse and return (client, handler).

    Priority:
    1) Use environment variables (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)
    2) If not set, optionally use hardcoded keys above (if you set them)
    3) Otherwise, return (None, None)
    """

    if os.environ.get("LANGFUSE_PUBLIC_KEY") is None and os.environ.get("LANGFUSE_SECRET_KEY") is None:
        return None, None

    try:
        langfuse = get_client()

        # Optionally, initialize the client with configuration options if you dont use environment variables (os.environ)
        # langfuse = Langfuse(public_key="pk-lf-...", secret_key="sk-lf-...")

        handler = CallbackHandler()
        return langfuse, handler
    except Exception as e:
        print(f"Langfuse initialization error: {e}")
        return None, None 
    # try:
    #     langfuse = Langfuse(public_key=LF_PUBLIC, secret_key=LF_SECRET, host=LF_HOST)
    #     handler = CallbackHandler()
    #     return langfuse, handler
    # except Exception as e:
    #     print(f"Langfuse initialization error: {e}")
    #     return None, None
