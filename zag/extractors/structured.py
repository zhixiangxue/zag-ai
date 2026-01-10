"""
Structured extractor using chak for type-safe extraction
"""

from typing import List, Dict, Sequence, Type

from pydantic import BaseModel

from .base import BaseExtractor


class StructuredExtractor(BaseExtractor):
    """
    Structured extractor that uses chak to extract structured information from units.
    
    Key design:
    - Uses chak (simple yet powerful LLM client with structured output)
    - Schema defined by developer (Pydantic BaseModel)
    - Automatic validation and type conversion
    - No wrapper layer - directly expands schema fields
    
    Args:
        llm_uri: LLM URI in format: provider/model
                 e.g., "openai/gpt-4o-mini", "bailian/qwen-plus"
        api_key: API key for the LLM provider
        schema: Pydantic BaseModel class (developer-defined)
        base_url: Optional custom base URL (for OpenAI-compatible providers)
    
    Example:
        >>> # 1. Define data structure (developer-defined)
        >>> class ProductInfo(BaseModel):
        ...     product_name: str
        ...     apr_min: float
        ...     loan_term_years: int
        >>> 
        >>> # 2. Create extractor
        >>> extractor = StructuredExtractor(
        ...     llm_uri="bailian/qwen-plus",
        ...     api_key="sk-xxx",
        ...     schema=ProductInfo
        ... )
        >>> 
        >>> # 3. Use
        >>> results = await extractor.aextract(units)
        >>> 
        >>> # 4. Access results (directly expanded, no wrapper)
        >>> print(units[0].metadata.custom["product_name"])
        >>> print(units[0].metadata.custom["apr_min"])
        >>> # "30-Year Fixed Rate Mortgage"
        >>> # 6.5
    """
    
    def __init__(
        self,
        llm_uri: str,
        api_key: str,
        schema: Type[BaseModel],
        base_url: str = None,
    ):
        self.llm_uri = llm_uri
        self.api_key = api_key
        self.schema = schema
        self.base_url = base_url
        
        # Import chak
        try:
            import chak
        except ImportError:
            raise ImportError(
                "chak is required for StructuredExtractor. "
                "Install it with: pip install chakpy"
            )
        
        # Create chak conversation
        # chak handles provider compatibility automatically
        self._conv = chak.Conversation(
            llm_uri,
            api_key=api_key,
            base_url=base_url
        )
    
    async def _extract_from_unit(self, unit) -> Dict:
        """Extract structured information from a single unit."""
        content = unit.content if hasattr(unit, 'content') else str(unit)
        
        # Build prompt with clear instructions
        # Use English for better LLM understanding
        prompt = f"""Extract structured information from the following text and return it in the exact format specified.

IMPORTANT:
- Extract information accurately based on the text content
- Maintain the same language as the input text for extracted values
- If information is not found, use appropriate default values or null
- Do not make up information that is not present in the text

Text to analyze:
{content}

Please extract the requested structured information from above."""
        
        try:
            # Use chak's structured output feature
            result = await self._conv.asend(prompt, returns=self.schema)
            
            # result is already a Pydantic model instance
            # Directly expand schema fields, no wrapper layer
            return result.model_dump()
        except Exception as e:
            print(f"Warning: Failed to extract structured data: {e}")
            import traceback
            traceback.print_exc()
            return {}
