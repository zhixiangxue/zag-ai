"""
Provider base class (internal use only)
"""

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """
    Base provider class for embedder implementations
    
    All provider implementations must inherit from this class
    and implement the required methods
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text into a vector
        
        Args:
            text: Input text to embed
            
        Returns:
            Vector representation of the text
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts into vectors
        
        Args:
            texts: List of input texts
            
        Returns:
            List of vector representations
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Get the embedding dimension
        
        Returns:
            Dimension of the embedding vectors
        """
        pass
