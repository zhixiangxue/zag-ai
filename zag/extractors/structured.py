"""
Structured extractor using instructor for type-safe extraction
"""

from typing import List, Dict, Sequence, Type

from pydantic import BaseModel

from .base import BaseExtractor


class StructuredExtractor(BaseExtractor):
    """
    Structured extractor that uses instructor to extract structured information from units.
    
    Key design:
    - Uses instructor (professional structured extraction library)
    - Schema defined by developer (Pydantic BaseModel)
    - Automatic retry and validation
    - No wrapper layer - directly expands schema fields
    
    Args:
        llm_uri: LLM URI in format: provider/model
                 e.g., "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"
        api_key: API key for the LLM provider
        schema: Pydantic BaseModel class (developer-defined)
        max_retries: Maximum retries on validation failure, default 3
    
    Example:
        >>> # 1. Define data structure (developer-defined)
        >>> class ProductInfo(BaseModel):
        ...     product_name: str
        ...     apr_min: float
        ...     loan_term_years: int
        >>> 
        >>> # 2. Create extractor
        >>> extractor = StructuredExtractor(
        ...     llm_uri="openai/gpt-4o-mini",
        ...     api_key="sk-xxx",
        ...     schema=ProductInfo
        ... )
        >>> 
        >>> # 3. Use
        >>> units = extractor(units)
        >>> 
        >>> # 4. Access results (directly expanded, no wrapper)
        >>> print(units[0].metadata["product_name"])
        >>> print(units[0].metadata["apr_min"])
        >>> # "30-Year Fixed Rate Mortgage"
        >>> # 6.5
    """
    
    def __init__(
        self,
        llm_uri: str,
        api_key: str,
        schema: Type[BaseModel],
        max_retries: int = 3,
    ):
        self.llm_uri = llm_uri
        self.api_key = api_key
        self.schema = schema
        self.max_retries = max_retries
        
        # Use instructor (supports provider/model URI)
        import instructor
        from openai import OpenAI
        
        # Try instructor's built-in provider support first
        try:
            self._client = instructor.from_provider(llm_uri, api_key=api_key)
            self._model = None  # Provider URI includes model
        except Exception as e:
            # Fallback: try OpenAI compatibility mode
            # This handles providers like bailian that use OpenAI-compatible APIs
            if "/" not in llm_uri:
                raise ValueError(f"Invalid LLM URI format: {llm_uri}. Expected 'provider/model'")
            
            provider, model = llm_uri.split("/", 1)
            provider = provider.lower()
            
            # Map known OpenAI-compatible providers to their base URLs
            openai_compatible_providers = {
                "bailian": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                # Add more OpenAI-compatible providers here
            }
            
            if provider in openai_compatible_providers:
                base_client = OpenAI(
                    api_key=api_key,
                    base_url=openai_compatible_providers[provider]
                )
                self._client = instructor.from_openai(base_client)
                self._model = model
            else:
                # Re-raise original error if not an OpenAI-compatible provider
                raise e
    
    async def _extract_from_unit(self, unit) -> Dict:
        """Extract structured information from a single unit."""
        content = unit.content if hasattr(unit, 'content') else str(unit)
        
        prompt = f"从以下文本中提取结构化信息：\n\n{content}"
        
        try:
            # Prepare create parameters
            create_params = {
                "response_model": self.schema,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_retries": self.max_retries,
            }
            
            # Add model if using bailian
            if self._model:
                create_params["model"] = self._model
            
            # Use instructor to extract structured data (sync call in async context)
            import asyncio
            result = await asyncio.to_thread(
                self._client.chat.completions.create,
                **create_params
            )
            
            # Directly expand schema fields, no wrapper layer
            return result.model_dump()
        except Exception as e:
            print(f"Warning: Failed to extract structured data: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    async def aextract(self, units: Sequence) -> List[Dict]:
        """Batch extract metadata from units."""
        import asyncio
        tasks = [self._extract_from_unit(unit) for unit in units]
        return await asyncio.gather(*tasks)
