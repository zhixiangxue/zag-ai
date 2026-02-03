"""Answer generator - produce answers from retrieved context.

This module defines a small utility that takes a user query and a list of
context items (text chunks with optional metadata) and uses an LLM to 
generate a final answer with basic citation support.
"""

from typing import List, Optional, Union, Dict, Any, TYPE_CHECKING

import chak

if TYPE_CHECKING:
    from . import Answer


class AnswerGenerator:
    """Generate answers from retrieved context using an LLM.

    The generator is intentionally minimal and focuses on a single task:
    turning a list of context items into a natural language answer.
    
    Context items can be:
    - Plain strings
    - Dicts with 'text' key and optional 'id', 'title' keys
    - Any object with 'text' attribute and optional 'node_id', 'title' attributes
    """

    def __init__(self, llm_uri: str, api_key: Optional[str] = None) -> None:
        # chak is required; imported at module load time
        self.llm_uri = llm_uri
        self.api_key = api_key
        self._conversation: Optional["chak.Conversation"] = None

    def _get_conversation(self):
        """Lazy-load chak conversation client."""

        if self._conversation is None:
            import chak as _chak

            self._conversation = _chak.Conversation(self.llm_uri, api_key=self.api_key)
        return self._conversation

    async def generate(
        self, 
        query: str, 
        contexts: List[Union[str, Dict[str, Any], Any]],
        prompt_template: Optional[str] = None
    ) -> "Answer":
        """Generate an answer from retrieved contexts.
        
        Args:
            query: User's question
            contexts: List of context items (strings, dicts, or objects with text attribute)
            prompt_template: Custom prompt template. Should contain {query} and {context} placeholders.
                           If None, uses a default general-purpose template.
        """

        if not contexts:
            from . import Answer
            return Answer(text="No relevant information was found.", citations=[])

        conv = self._get_conversation()

        # Extract text and metadata from various input formats
        context_parts: List[str] = []
        citations: List[str] = []
        
        for idx, item in enumerate(contexts):
            if isinstance(item, str):
                # Plain string
                context_parts.append(item)
                citations.append(str(idx))
            elif isinstance(item, dict):
                # Dict with 'text' and optional 'id', 'title'
                text = item.get('text', str(item))
                item_id = item.get('id') or item.get('node_id') or str(idx)
                title = item.get('title', '')
                citations.append(item_id)
                if title:
                    context_parts.append(f"[{item_id}] {title}:\n{text}")
                else:
                    context_parts.append(f"[{item_id}] {text}")
            else:
                # Object with attributes
                text = getattr(item, 'text', str(item))
                # Try common attribute names for ID
                item_id = (getattr(item, 'node_id', None) or 
                          getattr(item, 'id', None) or 
                          str(idx))
                title = getattr(item, 'title', '')
                # Support 'summary' fallback if text is empty
                if hasattr(item, 'summary') and item.summary:
                    text = item.summary
                    
                citations.append(item_id)
                if title:
                    context_parts.append(f"[{item_id}] {title}:\n{text}")
                else:
                    context_parts.append(f"[{item_id}] {text}")

        context = "\n\n".join(context_parts)

        if prompt_template:
            prompt = prompt_template.format(query=query, context=context)
        else:
            # Default template with basic constraints
            prompt = (
                "Answer the following question based strictly on the provided context. "
                "Do not use external knowledge or make assumptions beyond what is given.\n\n"
                f"Question: {query}\n\n"
                f"Context:\n{context}\n\n"
                "Answer:"
            )

        response = await conv.asend(prompt)
        answer_text = response.content.strip() if response and response.content else ""

        from . import Answer
        return Answer(text=answer_text, citations=citations)
