"""
File hashing utilities using xxhash

Provides fast file hashing for document integrity verification and deduplication.
"""

from pathlib import Path


def calculate_file_hash(file_path: Path | str, chunk_size: int = 8192) -> str:
    """
    Calculate xxhash of a file
    
    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        Hexadecimal hash string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If xxhash is not installed
        
    Example:
        >>> from pathlib import Path
        >>> hash_value = calculate_file_hash(Path("document.pdf"))
        >>> print(hash_value)  # e.g., "a1b2c3d4e5f6..."
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        import xxhash
    except ImportError:
        raise ImportError(
            "xxhash is required for file hashing. "
            "Install it with: pip install xxhash"
        )
    
    # Use xxh64 for good balance of speed and collision resistance
    hash_func = xxhash.xxh64()
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()
