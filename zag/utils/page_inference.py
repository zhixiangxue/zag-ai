"""
Page number inference utilities

Infer which pages a unit appears on after splitting,
using fuzzy position-based matching.
"""

import re
from typing import Optional
from difflib import SequenceMatcher

from ..schemas.base import BaseUnit, Page


def normalize_text(text: str) -> str:
    """
    Normalize text to reduce irrelevant differences
    
    Handles:
    - Multiple spaces → single space
    - Multiple newlines → single newline
    - Strip leading/trailing whitespace
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces → single space
    text = re.sub(r'\n+', '\n', text)     # Multiple newlines → single newline
    text = text.strip()
    
    return text


def find_best_match(
    needle: str,
    haystack: str,
    start_from: int = 0,
    threshold: float = 0.85
) -> Optional[tuple[int, int, float]]:
    """
    Find the best matching position of needle in haystack using fuzzy matching
    
    Strategy:
    1. For short texts (< 50 chars), use exact matching
    2. For longer texts, use signature-based fast matching
    3. Return position and similarity score
    
    Args:
        needle: Text to find (unit content)
        haystack: Text to search in (full document)
        start_from: Start search from this position (for sequential ordering)
        threshold: Minimum similarity score (0.0-1.0)
        
    Returns:
        (match_start, match_end, similarity) or None if no match found
    """
    needle_len = len(needle)
    
    # Strategy 1: Short text - use exact matching
    if needle_len < 50:
        idx = haystack.find(needle, start_from)
        if idx != -1:
            return (idx, idx + needle_len, 1.0)
        return None
    
    # Strategy 2: Extract signature (first 100 chars) for fast candidate location
    signature_len = min(100, needle_len)
    signature = needle[:signature_len]
    
    # Find all positions where signature might appear
    candidates = []
    pos = start_from
    
    while pos < len(haystack):
        idx = haystack.find(signature[:50], pos)  # Use first 50 chars for quick find
        if idx == -1:
            break
        candidates.append(idx)
        pos = idx + 1
        
        # Limit candidates to avoid excessive computation
        if len(candidates) >= 10:
            break
    
    if not candidates:
        return None
    
    # Strategy 3: For each candidate, compute similarity with window
    best_match = None
    best_similarity = 0.0
    
    window_size = int(needle_len * 1.2)  # Allow 20% length variation
    
    for candidate_pos in candidates:
        # Extract window around candidate
        window_start = max(0, candidate_pos - 50)
        window_end = min(len(haystack), candidate_pos + window_size + 50)
        window = haystack[window_start:window_end]
        
        # Compute similarity
        similarity = SequenceMatcher(None, needle, window).ratio()
        
        if similarity > best_similarity:
            best_similarity = similarity
            # Use candidate position as match boundaries
            match_end = min(window_end, candidate_pos + needle_len)
            best_match = (candidate_pos, match_end, similarity)
            
            # Early exit if very high similarity
            if similarity > 0.98:
                break
    
    # Check if similarity exceeds threshold
    if best_similarity >= threshold:
        return best_match
    
    return None


def infer_page_numbers(
    units: list[BaseUnit],
    pages: list[Page],
    similarity_threshold: float = 0.85
) -> None:
    """
    Infer page numbers for units using fuzzy position-based matching
    
    This function modifies units in-place, setting metadata.page_numbers.
    
    Algorithm:
    1. Build page position index (character offsets in full document)
    2. For each unit, find best matching position in full text
    3. Determine which pages overlap with that position
    4. Use sequential ordering constraint (next unit should be after previous)
    
    Args:
        units: List of units to enrich with page numbers
        pages: List of pages from the document
        similarity_threshold: Minimum similarity score (default: 0.85)
    """
    if not pages:
        return
    
    # Step 1: Build page position index
    page_positions = []
    current_pos = 0
    
    for page in pages:
        page_len = len(page.content)
        page_positions.append((
            current_pos,              # start
            current_pos + page_len,   # end
            page.page_number          # page number
        ))
        current_pos += page_len
    
    # Concatenate all pages into full text
    full_text = ''.join(p.content for p in pages)
    
    # IMPORTANT: Normalize full_text to match normalized unit content
    # This significantly improves matching after text processing by splitters
    full_text_normalized = normalize_text(full_text)
    
    # Step 2-3: For each unit, find position and infer pages
    last_found_pos = 0  # Sequential ordering constraint
    
    for unit in units:
        # Normalize unit content
        unit_text = normalize_text(unit.content)
        
        # Skip very short units (< 20 chars after normalization)
        if len(unit_text) < 20:
            unit.metadata.page_numbers = None
            continue
        
        # Find best matching position
        best_match = find_best_match(
            unit_text,
            full_text_normalized,  # Use normalized text for matching
            start_from=last_found_pos,
            threshold=similarity_threshold
        )
        
        if best_match is None:
            # No confident match found
            unit.metadata.page_numbers = None
            continue
        
        match_start, match_end, similarity = best_match
        last_found_pos = match_start  # Update for next unit
        
        # Step 4: Find overlapping pages
        overlapping_pages = []
        for page_start, page_end, page_num in page_positions:
            # Check overlap: not (end <= start or start >= end)
            if not (match_end <= page_start or match_start >= page_end):
                overlapping_pages.append(page_num)
        
        unit.metadata.page_numbers = overlapping_pages if overlapping_pages else None


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
