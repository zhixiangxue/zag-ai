"""
URI parsing utilities for embedder configuration

Supports two URI formats:
1. Simple format: provider/model
   - Uses default base_url from provider config
   - Example: "bailian/text-embedding-v3"

2. Full format: provider@base_url:model?params
   - Full control over base_url and parameters
   - Example: "bailian@https://dashscope.aliyuncs.com/compatible-mode/v1:text-embedding-v3?dimensions=1024"
   - Use "~" for default base_url: "bailian@~:text-embedding-v3"
"""

from typing import Dict, Any
from urllib.parse import urlencode, parse_qs

from .exceptions import URIError


def parse_embedder_uri(uri: str) -> Dict[str, Any]:
    """
    Parse an embedder URI into its components
    
    Supports two formats:
    1. Simple: "provider/model" (uses default base_url)
    2. Full: "provider@base_url:model?params" (full control)
    
    Args:
        uri: Embedder URI string
        
    Returns:
        Dictionary with keys: provider, base_url, model, params
        
    Raises:
        URIError: If URI format is invalid
        
    Examples:
        >>> parse_embedder_uri("bailian/text-embedding-v3")
        {'provider': 'bailian', 'base_url': None, 'model': 'text-embedding-v3', 'params': {}}
        
        >>> parse_embedder_uri("bailian@~:text-embedding-v3?dimensions=1024")
        {'provider': 'bailian', 'base_url': None, 'model': 'text-embedding-v3', 
         'params': {'dimensions': '1024'}}
    """
    if not uri or not isinstance(uri, str):
        raise URIError("URI must be a non-empty string")
    
    # Detect format based on presence of '@'
    if '@' in uri:
        # Full format: provider@base_url:model?params
        return _parse_full_format(uri)
    elif '/' in uri:
        # Simple format: provider/model
        return _parse_simple_format(uri)
    else:
        raise URIError(
            f"Invalid URI format: {uri}\n"
            f"Expected formats:\n"
            f"  - Simple: provider/model (e.g., 'bailian/text-embedding-v3')\n"
            f"  - Full: provider@base_url:model?params (e.g., 'bailian@~:text-embedding-v3')"
        )


def _parse_simple_format(uri: str) -> Dict[str, Any]:
    """
    Parse simple format URI: provider/model
    
    Args:
        uri: Simple format URI (e.g., "bailian/text-embedding-v3")
        
    Returns:
        Dictionary with provider, model, base_url=None, params={}
        
    Raises:
        URIError: If format is invalid
    """
    # Simple format should not have query parameters
    if '?' in uri:
        raise URIError(
            f"Simple format URI cannot contain query parameters: {uri}\n"
            f"Use full format for parameters: provider@base_url:model?params"
        )
    
    # Split by first '/'
    parts = uri.split('/', 1)
    if len(parts) != 2:
        raise URIError(f"Invalid simple format URI: {uri}\nExpected: provider/model")
    
    provider, model = parts
    
    # Validate provider and model
    if not provider or not model:
        raise URIError(f"Provider and model cannot be empty: {uri}")
    
    # Provider should not contain special characters
    if any(c in provider for c in '@:~?#/'):
        raise URIError(f"Invalid provider name: {provider}")
    
    return {
        'provider': provider,
        'base_url': None,  # Will use default from provider config
        'model': model,
        'params': {}
    }


def _parse_full_format(uri: str) -> Dict[str, Any]:
    """
    Parse full format URI: provider@base_url:model?params
    
    Args:
        uri: Full format URI
        
    Returns:
        Dictionary with provider, base_url, model, params
    """
    if not uri or not isinstance(uri, str):
        raise URIError("URI must be a non-empty string")

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
    base_url_part = None
    model = None
    
    # Case 1: base_url is full HTTP(S) URL
    if rest.startswith('http://') or rest.startswith('https://'):
        # Find the separator colon between URL and model
        protocol_end = rest.index('//') + 2
        rest_after_protocol = rest[protocol_end:]
        
        split_index = -1
        for i, char in enumerate(rest_after_protocol):
            if char == ':':
                # Check what comes after the colon
                next_part = rest_after_protocol[i+1:i+10]
                # If not a digit or /, it's the model separator
                if next_part and not next_part[0].isdigit() and not next_part[0] == '/':
                    split_index = protocol_end + i
                    break
        
        if split_index > 0:
            base_url_part = rest[:split_index]
            model = rest[split_index + 1:]
        else:
            # Fallback to last colon
            last_colon = rest.rfind(':')
            base_url_part = rest[:last_colon]
            model = rest[last_colon + 1:]
    
    # Case 2: base_url is ~ (default)
    elif rest.startswith('~:'):
        base_url_part = '~'
        model = rest[2:]  # Skip ~:
    
    # Case 3: base_url might be host:port format
    else:
        first_colon = rest.index(':')
        after_first_colon = rest[first_colon + 1:]
        
        # If colon is followed by digits, it's a port
        if after_first_colon and after_first_colon[0].isdigit():
            port_end = first_colon + 1
            while port_end < len(rest) and rest[port_end].isdigit():
                port_end += 1
            
            if port_end < len(rest) and rest[port_end] == ':':
                # Found colon after port
                base_url_part = rest[:port_end]
                model = rest[port_end + 1:]
            else:
                # Use last colon
                last_colon = rest.rfind(':')
                base_url_part = rest[:last_colon]
                model = rest[last_colon + 1:]
        else:
            # First colon is the separator
            base_url_part = rest[:first_colon]
            model = rest[first_colon + 1:]

    # Handle base_url: ~ means default/None
    base_url = None if base_url_part == "~" else base_url_part

    # Parse query parameters
    params = {}
    if query_string:
        try:
            parsed = parse_qs(query_string, keep_blank_values=False)
            for key, values in parsed.items():
                if len(values) == 1:
                    params[key] = values[0]
                else:
                    params[key] = values
        except Exception as e:
            raise URIError(f"Failed to parse query string: {e}")

    return {
        'provider': provider,
        'base_url': base_url,
        'model': model,
        'params': params
    }


def build_embedder_uri(
    provider: str,
    model: str,
    base_url: str | None = None,
    **params: Any
) -> str:
    """
    Build an embedder URI from components
    
    Args:
        provider: Provider name (e.g., "bailian", "openai")
        model: Model name (e.g., "text-embedding-v3")
        base_url: Custom base URL, or None for default (~)
        **params: Query parameters
        
    Returns:
        Formatted URI string
        
    Examples:
        >>> build_embedder_uri("bailian", "text-embedding-v3")
        'bailian@~:text-embedding-v3'
        
        >>> build_embedder_uri("bailian", "text-embedding-v3", dimensions=1024)
        'bailian@~:text-embedding-v3?dimensions=1024'
    """
    # Validate inputs
    if not provider or not isinstance(provider, str):
        raise URIError("Provider must be a non-empty string")
    if not model or not isinstance(model, str):
        raise URIError("Model must be a non-empty string")

    # Validate provider doesn't contain special characters
    if any(c in provider for c in '@:~?#'):
        raise URIError(f"Provider cannot contain special characters: {provider}")
    if any(c in model for c in '@~?#'):
        raise URIError(f"Model cannot contain special characters (@~?#): {model}")

    # Use ~ as placeholder for default base_url
    authority = "~" if base_url is None else base_url.rstrip('/')

    # Build URI: provider@authority:model
    uri = f"{provider}@{authority}:{model}"

    # Add query parameters
    if params:
        filtered_params = {k: v for k, v in params.items() if v is not None}
        if filtered_params:
            query_string = urlencode(filtered_params)
            uri = f"{uri}?{query_string}"

    return uri
