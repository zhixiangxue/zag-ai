"""
LLM-based passage selector

Uses LLM to extract relevant passages from long documents.
"""

from typing import Optional, List
from pydantic import BaseModel, Field

from ..base import BasePostprocessor
from ...schemas.base import BaseUnit


class ExcerptResult(BaseModel):
    """Result for a single unit"""
    unit_id: str = Field(description="Unit ID")
    excerpts: List[str] = Field(
        description="List of relevant excerpts (continuous text from original content)"
    )
    is_relevant: bool = Field(description="Whether this unit contains relevant content")


class ExtractionResults(BaseModel):
    """Wrapper for list of excerpt results"""
    results: List[ExcerptResult] = Field(description="List of extraction results for all units")


class LLMSelector(BasePostprocessor):
    """
    LLM-based passage selector
    
    Extracts relevant passages from each unit's content using LLM.
    Irrelevant units are removed from results.
    
    Workflow:
        1. Send query + all units to LLM
        2. LLM identifies relevant continuous passages in each unit
        3. Replace unit content with extracted passages
        4. Remove units with no relevant content
    
    Examples:
        >>> from zag.postprocessors import LLMSelector
        >>> 
        >>> selector = LLMSelector(llm_uri="ollama/qwen2.5:7b")
        >>> result = selector.process(query, units)
    """
    
    def __init__(
        self,
        llm_uri: str,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM selector
        
        Args:
            llm_uri: LLM URI (e.g., "ollama/qwen2.5:7b", "openai/gpt-4o")
            api_key: API key for cloud providers (optional, reads from env if not provided)
        """
        self.llm_uri = llm_uri
        self.api_key = api_key
        self._conversation = None
    
    def _get_conversation(self):
        """Lazy load conversation client"""
        if self._conversation is None:
            try:
                import chak
            except ImportError:
                raise ImportError(
                    "chak is required for LLMSelector. "
                    "Install it with: pip install chakpy"
                )
            
            # Create conversation
            self._conversation = chak.Conversation(self.llm_uri, api_key=self.api_key)
        
        return self._conversation
    
    def _build_prompt(self, query: str, units: List[BaseUnit]) -> str:
        """Build prompt for LLM"""
        prompt_parts = [
            "Analyze the following search results and extract relevant continuous passages from each document.",
            "",
            f"Question: {query}",
            "",
            "Search Results:",
            ""
        ]
        
        for unit in units:
            prompt_parts.append(f"[ID: {unit.unit_id}]")
            prompt_parts.append(str(unit.content))
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Requirements:",
            "1. For each document, identify continuous passages relevant to the question",
            "2. Passages must be extracted verbatim from the original text without any modifications",
            "3. A document can have multiple relevant passages",
            "4. If a document is completely irrelevant, set is_relevant to false",
            "5. Return each document's unit_id, list of excerpts, and is_relevant flag",
            "",
            "Note: Only extract truly relevant content. Do not force extraction of irrelevant passages."
        ])
        
        return "\n".join(prompt_parts)
    
    async def aprocess(
        self,
        query: str,
        units: List[BaseUnit]
    ) -> List[BaseUnit]:
        """
        Extract relevant passages from units (async version)
        
        Args:
            query: User's query
            units: Units to process
            
        Returns:
            Units with extracted passages (irrelevant units removed)
        """
        if not units:
            return []
        
        # Get conversation client
        conv = self._get_conversation()
        
        # Build prompt
        prompt = self._build_prompt(query, units)
        
        # Call LLM with structured output
        try:
            response = await conv.asend(
                prompt,
                returns=ExtractionResults  # Use wrapper instead of List[ExcerptResult]
            )
            
            # Handle None response
            if response is None:
                raise RuntimeError("LLM returned None. Check if the model supports structured output.")
            
            # Extract results list
            excerpt_results = response.results
            
        except Exception as e:
            raise RuntimeError(f"LLM selection failed: {e}")
        
        # Process results
        result_units = []
        
        # Create a mapping from unit_id to unit for fast lookup
        unit_map = {unit.unit_id: unit for unit in units}
        
        for excerpt_result in excerpt_results:
            # Skip irrelevant units
            if not excerpt_result.is_relevant or not excerpt_result.excerpts:
                continue
            
            # Find original unit
            original_unit = unit_map.get(excerpt_result.unit_id)
            if not original_unit:
                continue
            
            # Create new unit with extracted content
            new_unit = original_unit.model_copy()
            new_unit.content = "\n\n".join(excerpt_result.excerpts)
            result_units.append(new_unit)
        
        return result_units
    
    def process(
        self,
        query: str,
        units: List[BaseUnit]
    ) -> List[BaseUnit]:
        """
        Extract relevant passages from units (sync version)
        
        Args:
            query: User's query
            units: Units to process
            
        Returns:
            Units with extracted passages (irrelevant units removed)
        """
        import asyncio
        return asyncio.run(self.aprocess(query, units))
