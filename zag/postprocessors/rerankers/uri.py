"""
URI parsing utilities for reranker configuration

Supports two URI formats:
1. Simple format: provider/model
   Example: "sentence_transformers/cross-encoder-ms-marco"

2. Full format: provider@base_url:model?params
   Example: "cohere@~:rerank-english-v2.0?top_k=10"
"""

from typing import Dict, Any
from urllib.parse import parse_qs
from .exceptions import URIError


def parse_reranker_uri(uri: str) -> Dict[str, Any]:
    """
    Parse a reranker URI into its components
    
    Supports two formats:
    1. Simple: "provider/model"
    2. Full: "provider@base_url:model?params"
    
    Args:
        uri: Reranker URI string
        
    Returns:
        Dictionary with keys: provider, base_url, model, params
        
    Raises:
        URIError: If URI format is invalid
        
    Examples:
        >>> parse_reranker_uri("sentence_transformers/cross-encoder-ms-marco")
        {'provider': 'sentence_transformers', 'base_url': None, 'model': 'cross-encoder-ms-marco', 'params': {}}
        
        >>> parse_reranker_uri("cohere@~:rerank-v2.0?top_k=10")
        {'provider': 'cohere', 'base_url': None, 'model': 'rerank-v2.0', 
         'params': {'top_k': '10'}}
    """
    if not uri or not isinstance(uri, str):
        raise URIError("URI must be a non-empty string")
    
    # Detect format based on presence of '@'
    if '@' in uri:
        # Full format
        return _parse_full_format(uri)
    elif '/' in uri:
        # Simple format
        return _parse_simple_format(uri)
    else:
        raise URIError(
            f"Invalid URI format: {uri}\n"
            f"Expected formats:\n"
            f"  - Simple: provider/model (e.g., 'sentence_transformers/cross-encoder-ms-marco')\n"
            f"  - Full: provider@base_url:model?params (e.g., 'cohere@~:rerank-v2.0')"
        )


def _parse_simple_format(uri: str) -> Dict[str, Any]:
    """Parse simple format: provider/model"""
    if '?' in uri:
        raise URIError(
            f"Simple format URI cannot contain query parameters: {uri}\n"
            f"Use full format for parameters: provider@base_url:model?params"
        )
    
    parts = uri.split('/', 1)
    if len(parts) != 2:
        raise URIError(f"Invalid simple format URI: {uri}\nExpected: provider/model")
    
    provider, model = parts
    
    if not provider or not model:
        raise URIError(f"Provider and model cannot be empty: {uri}")
    
    return {
        'provider': provider,
        'base_url': None,
        'model': model,
        'params': {}
    }


def _parse_full_format(uri: str) -> Dict[str, Any]:
    """Parse full format: provider@base_url:model?params"""
    # Check for query parameters
    if '?' in uri:
        uri_part, query_string = uri.split('?', 1)
    else:
        uri_part = uri
        query_string = None
    
    # Split provider and rest
    if '@' not in uri_part:
        raise URIError(f"Invalid URI format: missing '@' separator in {uri}")
    
    provider, rest = uri_part.split('@', 1)
    
    if ':' not in rest:
        raise URIError(f"Invalid URI format: missing ':' separator in {uri}")
    
    # Parse base_url and model
    # Handle ~ as default base_url
    if rest.startswith('~:'):
        base_url = None
        model = rest[2:]
    else:
        # Split by last colon (simple approach)
        last_colon = rest.rfind(':')
        base_url_part = rest[:last_colon]
        model = rest[last_colon + 1:]
        base_url = None if base_url_part == '~' else base_url_part
    
    # Parse query parameters
    params = {}
    if query_string:
        try:
            parsed = parse_qs(query_string, keep_blank_values=False)
            for key, values in parsed.items():
                params[key] = values[0] if len(values) == 1 else values
        except Exception as e:
            raise URIError(f"Failed to parse query string: {e}")
    
    return {
        'provider': provider,
        'base_url': base_url,
        'model': model,
        'params': params
    }
