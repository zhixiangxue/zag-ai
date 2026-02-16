"""
Heading corrector for fixing heading formats in document content

Corrects heading formats in doc.content and page.content using extracted heading metadata
"""

import re
from typing import List, Dict

from ..base import BasePostprocessor
from ...schemas import BaseUnit
from ...schemas.pdf import PDF
from ...extractors.heading import HeadingExtractor


class HeadingCorrector(BasePostprocessor):
    """
    Correct heading formats in PDF document content
    
    This postprocessor:
    1. Extracts heading structure from PDF using HeadingExtractor
    2. Uses LLM to correct heading text (optional)
    3. Fixes heading markdown format in doc.content and page.content
    
    It converts incorrectly formatted text into proper markdown headings:
    Before: "SECTION 1 OVERVIEW"
    After:  "# SECTION 1 OVERVIEW"
    
    Note: This works on PDF documents, not on units
    Use this in document processing pipeline, not retrieval pipeline
    
    Args:
        llm_uri: LLM URI in format: provider/model (required if llm_correction=True)
        api_key: API key for the LLM provider (required if llm_correction=True)
        llm_correction: Whether to use LLM for heading text correction
    
    Example:
        >>> # Without LLM correction
        >>> corrector = HeadingCorrector()
        >>> pdf = corrector.correct_document(pdf)
        >>>
        >>> # With LLM correction
        >>> corrector = HeadingCorrector(
        ...     llm_uri="openai/gpt-4o-mini",
        ...     api_key="sk-xxx",
        ...     llm_correction=True
        ... )
        >>> pdf = await corrector.acorrect_document(pdf)
    """
    
    def __init__(
        self,
        llm_uri: str = None,
        api_key: str = None,
        llm_correction: bool = False
    ):
        """
        Initialize heading corrector
        
        Args:
            llm_uri: LLM URI (required if llm_correction=True)
            api_key: API key (required if llm_correction=True)
            llm_correction: Whether to use LLM for heading text correction
        """
        self.llm_correction = llm_correction
        self.llm_uri = llm_uri
        self.api_key = api_key
        
        if llm_correction and not llm_uri:
            raise ValueError("llm_uri is required when llm_correction=True")
    
    def process(self, query: str, units: List[BaseUnit]) -> List[BaseUnit]:
        """
        Postprocessor interface (not used for this corrector)
        
        This corrector works on PDF documents, not units.
        Use correct_document() method instead.
        """
        raise NotImplementedError(
            "HeadingCorrector works on PDF documents. "
            "Use correct_document(pdf: PDF) or acorrect_document(pdf: PDF) instead of process()"
        )
    
    def correct_document(self, pdf: PDF) -> PDF:
        """
        Correct heading formats in PDF document (sync version, no LLM correction)
        
        Args:
            pdf: PDF document with pages and content
            
        Returns:
            PDF document with corrected heading formats
        """
        if self.llm_correction:
            raise RuntimeError(
                "LLM correction requires async. Use: await corrector.acorrect_document(pdf)"
            )
        
        if not pdf.metadata or not pdf.metadata.source:
            raise ValueError("PDF must have metadata.source for heading extraction")
        
        # Extract headings from PDF file
        extractor = HeadingExtractor()
        headings = extractor.extract_from_pdf(pdf.metadata.source)
        
        if not headings:
            return pdf
        
        # Fix heading formats in page content
        self._correct_page_headings(pdf, headings)
        
        return pdf
    
    async def acorrect_document(self, pdf: PDF) -> PDF:
        """
        Correct heading formats in PDF document (async version, with optional LLM correction)
        
        Args:
            pdf: PDF document with pages and content
            
        Returns:
            PDF document with corrected heading formats
        """
        if not pdf.metadata or not pdf.metadata.source:
            raise ValueError("PDF must have metadata.source for heading extraction")
        
        # Extract headings from PDF file with optional LLM correction
        extractor = HeadingExtractor(
            llm_uri=self.llm_uri,
            api_key=self.api_key,
            llm_correction=self.llm_correction
        )
        headings = await extractor.aextract_from_pdf(pdf.metadata.source)
        
        if not headings:
            return pdf
        
        # Fix heading formats in page content
        self._correct_page_headings(pdf, headings)
        
        return pdf
    
    def _correct_page_headings(self, pdf: PDF, headings: List[Dict]):
        """Fix heading formats in page content and rebuild doc.content"""
        # Group headings by page
        headings_by_page = {}
        for h in headings:
            page_num = h['page']
            if page_num not in headings_by_page:
                headings_by_page[page_num] = []
            headings_by_page[page_num].append(h)
        
        # Process each page
        for page in pdf.pages:
            page_headings = headings_by_page.get(page.page_number, [])
            if not page_headings:
                continue
            
            # Sort by position (top to bottom)
            page_headings.sort(key=lambda x: x['y_from_top'])
            
            # Replace heading text with markdown format
            content = page.content
            for h in page_headings:
                # Skip if heading text is inside a table
                if self._is_text_in_table(content, h['text']):
                    continue
                
                # Normalize heading text: strip and compress multiple spaces
                normalized_heading = ' '.join(h['text'].split())
                
                # Escape special regex characters
                heading_pattern = re.escape(normalized_heading)
                
                # Create markdown heading
                markdown_heading = '#' * h['level'] + ' ' + normalized_heading
                
                # Pattern to match: optional existing markdown heading (#*) + heading text
                # This handles cases where Docling already output markdown headings
                # e.g., "## Assets and Reserves" should become "## Assets and Reserves" (corrected level)
                # not "## ### Assets and Reserves" (duplicated)
                pattern_with_prefix = r'^#{0,6}\s*' + heading_pattern
                
                # Try to replace with existing markdown prefix first (at line start)
                new_content, count = re.subn(
                    pattern_with_prefix, 
                    markdown_heading, 
                    content, 
                    count=1, 
                    flags=re.MULTILINE | re.IGNORECASE
                )
                
                if count == 0:
                    # No existing markdown prefix found, try plain text replacement
                    content = re.sub(
                        heading_pattern, 
                        markdown_heading, 
                        content, 
                        count=1, 
                        flags=re.IGNORECASE
                    )
                else:
                    content = new_content
            
            page.content = content
        
        # Rebuild doc.content from all pages
        pdf.content = "\n\n".join(page.content for page in pdf.pages)
    
    def _is_text_in_table(self, content: str, text: str) -> bool:
        """
        Check if text appears inside a table (markdown or HTML)
        
        Args:
            content: Page content
            text: Text to check
            
        Returns:
            True if text is inside a table, False otherwise
        """
        # Find all occurrences of the text
        text_escaped = re.escape(text)
        
        for match in re.finditer(text_escaped, content):
            pos = match.start()
            
            # Check if inside markdown table
            # Look backwards for table row markers (|)
            before = content[:pos]
            after = content[pos:]
            
            # Check markdown table: text between | ... | on same line
            line_start = before.rfind('\n') + 1
            line_end = after.find('\n')
            if line_end == -1:
                line_end = len(after)
            
            current_line = content[line_start:pos + line_end]
            
            # Markdown table: has | on both sides
            if '|' in current_line:
                before_text = content[line_start:pos]
                after_text = content[pos:line_start + len(current_line)]
                if '|' in before_text and '|' in after_text:
                    return True
            
            # Check HTML table: text inside <table>...</table>
            # Find nearest <table> and </table> tags
            table_start = before.rfind('<table')
            table_end = after.find('</table>')
            
            if table_start != -1 and table_end != -1:
                # Check if there's a closing </table> between table_start and pos
                close_before = before[table_start:].find('</table>')
                if close_before == -1 or close_before > (pos - table_start):
                    return True
        
        return False
