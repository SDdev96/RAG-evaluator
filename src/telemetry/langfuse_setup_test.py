"""
Langfuse setup utilities using Object-Oriented Programming paradigm.

Usage:
    from src.telemetry.langfuse_setup import LangfuseManager
    
    # Initialize manager
    manager = LangfuseManager()
    
    # Use with context manager (recommended)
    with manager:
        response = manager.invoke_model(model, prompt, extra_config={"temperature": 0.7})
    
    # Or manual initialization/cleanup
    manager.initialize()
    try:
        response = manager.invoke_model(model, prompt)
    finally:
        manager.cleanup()
"""

import os
import logging
from typing import Optional, Any, Dict, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dotenv import load_dotenv

# External SDK
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

from config.config import LangfuseConfig

# Load .env to allow LANGFUSE_* variables
load_dotenv()


@dataclass
class LangfuseConfig:
    """Configuration class for Langfuse settings."""
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    release: Optional[str] = None
    host: str = "https://cloud.langfuse.com"
    debug: bool = False
    
    @classmethod
    def from_environment(cls) -> 'LangfuseConfig':
        """Create configuration from environment variables."""
        return cls(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            release=os.getenv("LANGFUSE_RELEASE", None),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            debug=os.getenv("LANGFUSE_DEBUG", "false").lower() == "true"
        )
    
    def is_valid(self) -> bool:
        """Check if the configuration has required keys."""
        return bool(self.public_key and self.secret_key)


class LangfuseInitializationError(Exception):
    """Custom exception for Langfuse initialization errors."""
    pass


class BaseLangfuseManager(ABC):
    """Abstract base class for Langfuse management."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize Langfuse client and handler."""
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if Langfuse is properly initialized."""
        pass

    @abstractmethod
    def get_client(self) -> Optional[Langfuse]:
        """Get the Langfuse client instance."""
        pass
    
    @abstractmethod
    def get_handler(self) -> Optional[CallbackHandler]:
        """Get the Langfuse callback handler."""
        pass


class LangfuseManager(BaseLangfuseManager):
    """
    Main class for managing Langfuse client and callback handler.
    
    Provides initialization, configuration, and model invocation utilities
    with proper error handling and logging.
    """
    
    def __init__(self, config: LangfuseConfig):
        """
        Initialize LangfuseManager.
        
        Args:
            config: Langfuse configuration. If None, loads from environment.
            logger: Logger instance. If None, creates a default logger.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self._client: Optional[Langfuse] = None
        self._handler: Optional[CallbackHandler] = None
        self._initialized = False

        self.initialize()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger for the manager."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if self.config.debug else logging.INFO)
        return logger
    
    def initialize(self) -> bool:
        """
        Initialize Langfuse client and callback handler.
        
        Returns:
            True if initialization was successful, False otherwise.
            
        Raises:
            LangfuseInitializationError: If initialization fails with invalid config.
        """
        if self._initialized:
            self.logger.debug("Langfuse already initialized")
            return True
        
        if not self.config.is_valid():
            error_msg = "Missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY in configuration"
            self.logger.warning(error_msg)
            raise LangfuseInitializationError(error_msg)
        
        try:
            # Set environment variables for langfuse client
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.config.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.config.secret_key
            os.environ["LANGFUSE_HOST"] = self.config.host
            os.environ["LANGFUSE_RELEASE"] = self.config.release
            
            self._client = get_client()
            self._handler = CallbackHandler()
            self._initialized = True
            
            self.logger.info("Langfuse initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Langfuse initialization failed: {e}"
            self.logger.error(error_msg)
            raise LangfuseInitializationError(error_msg) from e
    
    def is_initialized(self) -> bool:
        """Check if Langfuse is properly initialized."""
        return self._initialized and self._client is not None and self._handler is not None
    
    def get_client(self) -> Optional[Langfuse]:
        """Get the Langfuse client instance."""
        return self._client
    
    def get_handler(self) -> Optional[CallbackHandler]:
        """Get the Langfuse callback handler."""
        return self._handler
           
    def invoke_model_with_langchain(
        self, 
        model: Any, 
        prompt: Union[str, Dict[str, Any]], 
        extra_config: Optional[Dict[str, Any]] = None,
        auto_initialize: bool = True
    ) -> Any:
        """
        Invoke a LangChain model with Langfuse callback handler.
        
        Args:
            model: LangChain compatible model instance.
            prompt: Input for the model (string, dict, or compatible structure).
            extra_config: Optional additional configuration to merge.
            auto_initialize: Whether to automatically initialize if not already done.
        
        Returns:
            The model's response.
            
        Raises:
            LangfuseInitializationError: If Langfuse is not initialized and auto_initialize is False.
            ValueError: If handler is None after initialization attempts.
        """
        if not self.is_initialized():
            if auto_initialize:
                if not self.initialize():
                    raise LangfuseInitializationError("Failed to auto-initialize Langfuse")
            else:
                raise LangfuseInitializationError("Langfuse not initialized. Call initialize() first or set auto_initialize=True")
        
        if self._handler is None:
            raise ValueError("Langfuse handler is None after initialization")
        
        # Build configuration with callbacks
        config = dict(extra_config or {})
        callbacks = list(config.get("callbacks", []))
        callbacks.append(self._handler)
        config["callbacks"] = callbacks
        
        self.logger.debug(f"Invoking model with Langfuse handler. Config: {config}")
        
        try:
            response = model.invoke(prompt, config=config)
            self.logger.debug("Model invocation completed successfully")
            return response
        except Exception as e:
            self.logger.error(f"Model invocation failed: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up resources and reset state."""
        try:
            if self._handler:
                # Flush any pending traces
                if hasattr(self._handler, 'flush'):
                    self._handler.flush()
            
            if self._client:
                # Flush client if it has the method
                if hasattr(self._client, 'flush'):
                    self._client.flush()
            
            self.logger.info("Langfuse cleanup completed")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
        finally:
            self._client = None
            self._handler = None
            self._initialized = False
    
    def __enter__(self):
        """Context manager entry.
        
        Initializes Langfuse client and handler if not already initialized.
        
        Returns:
            LangfuseManager: The initialized LangfuseManager instance.
        """
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
