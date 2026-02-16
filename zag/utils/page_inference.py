"""
Page number inference utilities

Infer which pages a unit appears on after splitting,
using sequential ordering + signature matching.
"""

from typing import Optional
from difflib import SequenceMatcher

from ..schemas import BaseUnit, Page


def fuzzy_find_start(
    signature: str,
    haystack: str,
    start_from: int = 0,
    threshold: float = 0.80
) -> Optional[int]:
    """
    Find the start position of signature in haystack using fuzzy matching.
    
    Uses sliding window with similarity scoring.
    For efficiency, only searches in a reasonable range.
    
    Args:
        signature: Text to find (e.g., unit.content[:300])
        haystack: Text to search in (full document content)
        start_from: Start search from this position (sequential ordering)
        threshold: Minimum similarity ratio (0.0-1.0)
        
    Returns:
        Best matching position, or None if not found
    """
    sig_len = len(signature)
    haystack_len = len(haystack)
    
    # If signature is longer than remaining text, truncate
    if start_from + sig_len > haystack_len:
        available = haystack_len - start_from
        if available < 50:
            return None
        signature = signature[:available]
        sig_len = len(signature)
    
    # Step 1: Try exact match first (most efficient)
    # Use first 30 chars for quick exact search
    quick_sig = signature[:30] if len(signature) > 30 else signature
    if len(quick_sig) >= 10:
        pos = start_from
        while pos < haystack_len:
            idx = haystack.find(quick_sig, pos)
            if idx == -1:
                break
            # Verify with longer signature
            candidate = haystack[idx:idx + sig_len]
            if len(candidate) == sig_len:
                score = SequenceMatcher(None, signature, candidate).ratio()
                if score >= threshold:
                    return idx
            pos = idx + 1
            # Limit iterations
            if pos > start_from + 50000:
                break
    
    # Step 2: Sliding window with smaller step for better coverage
    best_pos = None
    best_score = threshold
    
    # Use smaller step for better coverage
    step = max(1, min(50, sig_len // 20))
    
    search_end = min(haystack_len - sig_len + 1, start_from + 100000)
    
    for pos in range(start_from, search_end, step):
        candidate = haystack[pos:pos + sig_len]
        score = SequenceMatcher(None, signature, candidate).ratio()
        
        if score > best_score:
            best_score = score
            best_pos = pos
            
            if score > 0.95:
                break
    
    return best_pos


def infer_page_numbers(
    units: list[BaseUnit],
    pages: list[Page],
    full_content: str = None,
    signature_len: int = 300,
    similarity_threshold: float = 0.70
) -> None:
    """
    Infer page numbers for units using sequential ordering + signature matching.
    
    Core algorithm:
    1. Build page offset index (page_number -> (start, end))
    2. For each unit in order:
       - Use unit.content[:signature_len] to find start position
       - end = start + len(unit.content)
       - Map (start, end) to page numbers
       - Update last_end_pos for next unit (sequential constraint)
    
    This modifies units in-place, setting metadata.page_numbers.
    
    Args:
        units: List of units to enrich with page numbers
        pages: List of pages from the document
        full_content: Full document content (if None, will be built from pages)
        signature_len: Length of signature for matching (default: 300)
        similarity_threshold: Minimum similarity for fuzzy matching (default: 0.70)
    """
    if not pages or not units:
        return
    
    # Use provided full_content, or build from pages
    if full_content is None:
        full_content = "\n".join(p.content or "" for p in pages)
    
    # Step 1: Build page offset index
    # Check if pages already have span info (from MinerUReader)
    page_positions = []
    has_span_info = all(
        p.metadata and hasattr(p.metadata, 'get') and p.metadata.get('span') 
        for p in pages
    )
    
    if has_span_info:
        # Use pre-computed span from Page.metadata
        for page in pages:
            span = page.metadata.get('span')
            if span:
                page_positions.append((span[0], span[1], page.page_number))
    else:
        # Fall back to searching for page content in full_content
        current_pos = 0
        
        for page in pages:
            page_content = page.content or ""
            if not page_content.strip():
                page_positions.append((current_pos, current_pos, page.page_number))
                continue
            
            page_sig = page_content[:50] if len(page_content) > 50 else page_content
            pos = full_content.find(page_sig, current_pos)
            
            if pos != -1:
                page_positions.append((pos, pos + len(page_content), page.page_number))
                current_pos = pos + 1
            else:
                page_positions.append((current_pos, current_pos + len(page_content), page.page_number))
                current_pos += len(page_content) + 1
    
    # Step 2: Process units sequentially
    last_end_pos = 0  # Sequential ordering constraint
    
    for unit in units:
        if not unit.content:
            unit.metadata.page_numbers = None
            continue
            
        content_len = len(unit.content)
            
        # Check if unit has known position (e.g., from parser)
        if unit.metadata.span:
            start, end = unit.metadata.span
        else:
            # Fall back to fuzzy matching
            # Use signature for matching
            sig = unit.content[:signature_len] if content_len > signature_len else unit.content
                
            # Skip very short units (less than 10 chars)
            if len(sig.strip()) < 10:
                unit.metadata.page_numbers = None
                continue
                
            # Try to find start position with multiple strategies
            start = None
                
            # Strategy 1: With sequential constraint
            start = fuzzy_find_start(sig, full_content, start_from=last_end_pos, threshold=similarity_threshold)
                
            # Strategy 2: Without sequential constraint (fallback)
            if start is None:
                start = fuzzy_find_start(sig, full_content, start_from=0, threshold=similarity_threshold)
                
            # Strategy 3: Try with smaller signature (more tolerant)
            if start is None and len(sig) > 100:
                smaller_sig = sig[:100]
                start = fuzzy_find_start(smaller_sig, full_content, start_from=0, threshold=0.60)
                
            if start is None:
                # Could not find
                unit.metadata.page_numbers = None
                continue
                
            # Calculate end position
            end = start + content_len
                
            # Update sequential constraint for next unit
            last_end_pos = max(last_end_pos, start + 1)
        
        # Step 3: Find overlapping pages
        overlapping_pages = []
        for page_start, page_end, page_num in page_positions:
            # Check overlap: not (content ends before page OR content starts after page)
            if not (end <= page_start or start >= page_end):
                overlapping_pages.append(page_num)
        
        unit.metadata.page_numbers = sorted(overlapping_pages) if overlapping_pages else None


def get_page_numbers_display(unit: BaseUnit) -> str:
    """
    Get human-readable page numbers string
    
    Args:
        unit: Unit to get page numbers from
        
    Returns:
        Display string like "p1", "p1-2", "p1,3,5", or "N/A"
    """
    if not hasattr(unit.metadata, 'page_numbers') or unit.metadata.page_numbers is None:
        return "N/A"
    
    pages = unit.metadata.page_numbers
    if not pages:
        return "N/A"
    
    if len(pages) == 1:
        return f"p{pages[0]}"
    
    # Check if consecutive
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"p{pages[0]}-{pages[-1]}"
    
    # Non-consecutive
    return "p" + ",".join(str(p) for p in pages)
