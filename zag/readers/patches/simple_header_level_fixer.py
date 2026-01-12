"""
Simple Header Level Fixer - Fix markdown header levels based on content patterns

🔧 TEMPORARY PATCH: This module provides a workaround for Docling Issue #1023
   (https://github.com/docling-project/docling/issues/1023)
   
   Docling currently exports all headers as H2 (##), losing the hierarchical structure.
   This simple rule-based fixer infers correct header levels from content patterns:
   
   1. Document keywords (CHAPTER, APPENDIX, etc.) → H1
   2. Numbering patterns (1.1 → H2, 1.1.1 → H3, etc.)
   3. Letter prefixes (A. → H3, AA. → H4)
   
   ⚠️  TODO: Remove this patch when Docling fixes the issue upstream.
   
   Note: This is a simple implementation. For more advanced scenarios,
   consider using LLM-based or ML-based approaches.
"""

import re
from typing import Optional, Callable


class SimpleHeaderLevelFixer:
    """
    Fix markdown header levels using simple rule-based inference
    
    This is a lightweight solution for fixing Docling's header level issue.
    Uses pattern matching on header text to infer correct hierarchy levels.
    
    Args:
        custom_rules: Optional custom rule function that takes header text
                     and returns level (1-6). If provided, overrides default rules.
    
    Example:
        >>> fixer = SimpleHeaderLevelFixer()
        >>> fixed_markdown = fixer.process(markdown_content)
        
        >>> # With custom rules
        >>> def my_rules(text: str) -> int:
        ...     if "PART" in text:
        ...         return 1
        ...     return 2
        >>> fixer = SimpleHeaderLevelFixer(custom_rules=my_rules)
    """
    
    def __init__(self, custom_rules: Optional[Callable[[str], int]] = None):
        """
        Initialize header level fixer
        
        Args:
            custom_rules: Optional function to infer header level from text
        """
        self.custom_rules = custom_rules
    
    def process(self, markdown_content: str) -> str:
        """
        Fix header levels in markdown content
        
        Args:
            markdown_content: Original markdown with incorrect header levels
            
        Returns:
            Fixed markdown with correct header levels
        """
        lines = markdown_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Check if it's a header line
            header_match = re.match(r'^(#+)\s+(.+)$', line)
            if not header_match:
                fixed_lines.append(line)
                continue
            
            header_text = header_match.group(2).strip()
            
            # Infer correct level
            if self.custom_rules:
                level = self.custom_rules(header_text)
            else:
                level = self._infer_level(header_text)
            
            # Reconstruct header with correct level
            fixed_lines.append(f"{'#' * level} {header_text}")
        
        return '\n'.join(fixed_lines)
    
    def _infer_level(self, header_text: str) -> int:
        """
        Infer header level from text using default rules
        
        Rules (in priority order):
        1. Document keywords: CHAPTER, APPENDIX, TABLE OF CONTENTS, GLOSSARY → H1
        2. Numbering patterns:
           - X.Y.Z.W (e.g., "1.2.3.4") → H4
           - X.Y.Z (e.g., "1.2.3") → H3
           - X.Y (e.g., "1.2") → H2
        3. Letter prefixes:
           - Single letter: "A.", "B.", "C." → H3
           - Double letter: "AA.", "AB." → H4
        4. SECTION keyword → H2
        5. Default → H2
        
        Args:
            header_text: The header text content
            
        Returns:
            Header level (1-6)
        """
        text = header_text.strip()
        text_upper = text.upper()
        
        # Level 1: Top-level document keywords
        h1_keywords = [
            'CHAPTER',
            'APPENDIX',
            'TABLE OF CONTENTS',
            'GLOSSARY',
            'ACRONYMS USED',
            'REFERENCES',
            'BIBLIOGRAPHY',
        ]
        
        if any(keyword in text_upper for keyword in h1_keywords):
            return 1
        
        # Check numbering patterns (most specific first)
        # X.Y.Z.W pattern (e.g., "1.2.3.4 Title") → H4
        if re.match(r'^\d+\.\d+\.\d+\.\d+\s', text):
            return 4
        
        # X.Y.Z pattern (e.g., "1.2.3 Title") → H3
        if re.match(r'^\d+\.\d+\.\d+\s', text):
            return 3
        
        # X.Y pattern (e.g., "1.2 Title") → H2
        if re.match(r'^\d+\.\d+\s', text):
            return 2
        
        # Level 3: Letter prefixes
        # Single letter: "A.", "B.", "C." → H3
        if re.match(r'^[A-Z]\.\s', text):
            return 3
        
        # Double letter: "AA.", "AB." → H4
        if re.match(r'^[A-Z]{2}\.\s', text):
            return 4
        
        # Level 2: SECTION keyword
        if 'SECTION' in text_upper and text_upper.startswith('SECTION'):
            return 2
        
        # Default to H2 for unknown patterns
        return 2
    
    def __repr__(self) -> str:
        """String representation"""
        rule_type = "custom" if self.custom_rules else "default"
        return f"SimpleHeaderLevelFixer(rules={rule_type})"
