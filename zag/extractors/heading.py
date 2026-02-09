"""
Heading extractor for PDF documents

Extracts heading structure from PDF using layout analysis + LLM correction
"""

from typing import Dict, List
from pathlib import Path
from pydantic import BaseModel, Field

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextContainer, LTChar
except ImportError:
    raise ImportError("pdfminer.six is required. Install: pip install pdfminer.six")

from .base import BaseExtractor


DEFAULT_CORRECTION_TEMPLATE = """You are a professional document processing expert specializing in information architecture and content restructuring.

Your task: Analyze and reconstruct the extracted heading structure to create a logical, hierarchical, and navigable outline.

**CRITICAL PRINCIPLE: CONSERVATIVE FILTERING - When in doubt, REMOVE IT**
- Precision > Recall: It's acceptable to miss some headings, but every kept heading MUST be genuine
- If you're not confident a text is a real heading (>80% certainty), EXCLUDE it from output
- Better to have an incomplete but accurate outline than a polluted one

Original headings (extracted from PDF):
{headings_json}

Core Principles (MUST FOLLOW):

1. Filter False Positives (STRICT):
   - REMOVE content that is NOT a real heading:
     * Page numbers, headers/footers, dates, standalone numbers
     * URLs, email addresses, phone numbers
     * Decorative text or formatting artifacts (e.g., "continued...", "----", "...")
     * Table headers or column labels (e.g., "Rate", "Amount", "Description")
     * Repeated elements across pages (likely page templates)
     * Short standalone words (<5 chars) unless clearly a section marker
     * Generic labels like "Table 1", "Figure 2", "Appendix"
   - KEEP only legitimate section/chapter/topic headings that represent document structure

2. Confidence Threshold:
   - For each heading, mentally rate confidence (0-100%)
   - Only keep headings with >80% confidence
   - Signs of LOW confidence (should remove):
     * Appears in unusual location (page margins, bottom of page)
     * Lacks semantic meaning (just numbers, codes, or abbreviations)
     * Seems like data value rather than label
     * Appears in middle of a sentence or paragraph flow

3. Merge & Deduplicate:
   - Merge consecutive identical headings on the same page into one
   - Reassemble fragmented headings split across lines (e.g., "CHAPTER 1:" + "REQUIREMENTS" → "CHAPTER 1: REQUIREMENTS")

4. Rebuild Hierarchy (CRITICAL):
   - Identify clues: numbering (e.g., CHAPTER 1, 1.1, A.), font size, keywords (SECTION, ATTACHMENT)
   - Establish parent-child relationships based on numbering patterns
   - Correct misclassified heading levels

5. Normalize Format:
   - Fix OCR errors (e.g., "SECT10N" → "SECTION", "l" → "1")
   - Correct obvious typos and formatting issues
   - Preserve original language (do NOT translate)
   - Maintain semantic meaning

6. Output Requirements:
   - Return corrected headings in the same JSON structure
   - If a heading is determined to be invalid or low confidence, EXCLUDE it from output
   - Preserve page numbers and metadata for valid headings only

Return the corrected headings:"""


class HeadingList(BaseModel):
    """Corrected heading list structure"""
    headings: List[Dict] = Field(description="Corrected headings with same structure as input")


class HeadingExtractor(BaseExtractor):
    """
    Extract heading structure from PDF documents with optional LLM correction
    
    Uses pdfminer.six layout analysis to detect headings based on:
    - Font size relative to body text
    - Font style (bold, font family)
    - Text formatting (all caps)
    
    Returns heading metadata including:
    - page: Page number
    - level: Heading level (1-6)
    - text: Heading text (optionally corrected by LLM)
    - position: Vertical position on page
    - font_info: Font attributes
    
    Args:
        llm_uri: LLM URI in format: provider/model (required if llm_correction=True)
        api_key: API key for the LLM provider (required if llm_correction=True)
        llm_correction: Whether to use LLM for heading text correction
    
    Example:
        >>> # Basic extraction without LLM
        >>> extractor = HeadingExtractor()
        >>> headings = extractor.extract_from_pdf("file.pdf")
        >>>
        >>> # With LLM correction
        >>> extractor = HeadingExtractor(
        ...     llm_uri="openai/gpt-4o-mini",
        ...     api_key="sk-xxx",
        ...     llm_correction=True
        ... )
        >>> headings = await extractor.aextract_from_pdf("file.pdf")
    """
    
    def __init__(
        self,
        llm_uri: str = None,
        api_key: str = None,
        llm_correction: bool = False
    ):
        """
        Initialize heading extractor
        
        Args:
            llm_uri: LLM URI (required if llm_correction=True)
            api_key: API key (required if llm_correction=True)
            llm_correction: Whether to use LLM for heading text correction
        """
        self.llm_correction = llm_correction
        self.llm_uri = llm_uri
        self.api_key = api_key
        
        if llm_correction:
            if not llm_uri:
                raise ValueError("llm_uri is required when llm_correction=True")
            
            # Initialize LLM client
            import chak
            self._conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def _extract_from_unit(self, unit) -> Dict:
        """Not used - this extractor works on document level"""
        raise NotImplementedError("HeadingExtractor works on PDF document, not units")
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract headings from PDF file (sync version, no LLM correction)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of heading dictionaries with metadata
        """
        if self.llm_correction:
            raise RuntimeError(
                "LLM correction requires async. Use: await extractor.aextract_from_pdf()"
            )
        
        return self._extract_headings_raw(pdf_path)
    
    async def aextract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract headings from PDF file (async version, with optional LLM correction)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of heading dictionaries with metadata (corrected if llm_correction=True)
        """
        headings = self._extract_headings_raw(pdf_path)
        
        if self.llm_correction and headings:
            headings = await self._llm_correct_headings(headings)
        
        return headings
    
    def _extract_headings_raw(self, pdf_path: str) -> List[Dict]:
        """
        Extract raw headings from PDF using layout analysis
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of heading dictionaries
        """
        if not pdf_path:
            raise ValueError("PDF path must be provided")
        
        # Configure layout analysis
        laparams = LAParams(
            line_overlap=0.5,
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            boxes_flow=0.5,
            detect_vertical=False,
            all_texts=False
        )
        
        # Extract all text elements with font info
        all_elements = []
        
        for page_num, page in enumerate(extract_pages(pdf_path, laparams=laparams), start=1):
            page_height = page.height
            
            for element in page:
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    if not text:
                        continue
                    
                    font_info = self._extract_font_info(element)
                    if not font_info:
                        continue
                    
                    y_from_top = page_height - element.y1
                    
                    all_elements.append({
                        'page': page_num,
                        'text': text,
                        'font_name': font_info['font_name'],
                        'font_size': font_info['font_size'],
                        'is_bold': font_info['is_bold'],
                        'x0': element.x0,
                        'y0': element.y0,
                        'x1': element.x1,
                        'y1': element.y1,
                        'width': element.width,
                        'height': element.height,
                        'y_from_top': y_from_top,
                    })
        
        if not all_elements:
            return []
        
        # Analyze font characteristics
        from collections import Counter
        font_size_counter = Counter([e['font_size'] for e in all_elements])
        normal_size = font_size_counter.most_common(1)[0][0]
        
        # Identify headings
        headings = []
        
        for elem in all_elements:
            text = elem['text']
            
            # Basic filters
            if len(text) < 2 or len(text) > 200:
                continue
            
            # Check if likely a heading
            size_ratio = elem['font_size'] / normal_size
            is_bold = elem['is_bold']
            is_short = len(text.split()) <= 15
            is_all_caps = text.isupper() and len(text.strip()) > 3
            
            # Criteria: Bold OR size > 1.1x normal, AND short enough
            if (is_bold or size_ratio > 1.1 or is_all_caps) and is_short:
                # Determine level by multiple factors
                if size_ratio >= 1.5:
                    level_by_size = 1
                elif size_ratio >= 1.3:
                    level_by_size = 2
                elif size_ratio >= 1.15:
                    level_by_size = 3
                elif size_ratio >= 1.05:
                    level_by_size = 4
                else:
                    level_by_size = 5
                
                # Adjust by text style
                if is_all_caps:
                    level_by_style = max(1, level_by_size - 1)
                elif is_bold:
                    level_by_style = level_by_size
                else:
                    level_by_style = min(6, level_by_size + 1)
                
                level = min(level_by_size, level_by_style)
                
                headings.append({
                    'page': elem['page'],
                    'text': text,
                    'level': level,
                    'font_name': elem['font_name'],
                    'font_size': elem['font_size'],
                    'size_ratio': round(size_ratio, 2),
                    'is_bold': is_bold,
                    'is_all_caps': is_all_caps,
                    'y_from_top': elem['y_from_top'],
                    'x0': elem['x0'],
                    'x1': elem['x1'],
                    'width': elem['width'],
                    'height': elem['height'],
                })
        
        # Merge horizontal headings on same line
        headings = self._merge_horizontal_headings(headings)
        
        # Sort by page and position
        headings.sort(key=lambda x: (x['page'], x['y_from_top']))
        
        return headings
    
    async def _llm_correct_headings(self, headings: List[Dict]) -> List[Dict]:
        """Use LLM to correct heading text (fix OCR errors, formatting issues)"""
        import json
        
        # Prepare minimal heading data for LLM (only what's needed for correction)
        headings_for_llm = [
            {
                'page': h['page'],
                'level': h['level'],
                'text': h['text'],
                'is_all_caps': h['is_all_caps']
            }
            for h in headings
        ]
        
        prompt = DEFAULT_CORRECTION_TEMPLATE.format(
            headings_json=json.dumps(headings_for_llm, ensure_ascii=False, indent=2)
        )
        
        try:
            response = await self._conv.asend(prompt, returns=HeadingList)
            
            if response is None or not response.headings:
                return headings
            
            # Merge corrected text back to original headings
            corrected_headings = headings.copy()
            for i, corrected in enumerate(response.headings):
                if i < len(corrected_headings):
                    corrected_headings[i]['text'] = corrected.get('text', headings[i]['text'])
            
            return corrected_headings
        
        except Exception as e:
            print(f"Warning: LLM heading correction failed: {e}")
            return headings
    
    def _extract_font_info(self, element) -> Dict:
        """Extract font information from text element"""
        for item in element:
            if hasattr(item, '__iter__'):
                for subitem in item:
                    if isinstance(subitem, LTChar):
                        font_name = subitem.fontname
                        font_size = round(subitem.height, 1)
                        is_bold = 'bold' in font_name.lower() or 'black' in font_name.lower()
                        
                        return {
                            'font_name': font_name,
                            'font_size': font_size,
                            'is_bold': is_bold
                        }
        return None
    
    def _merge_horizontal_headings(
        self, 
        headings: List[Dict], 
        y_tolerance: float = 3, 
        x_gap_max: float = 50
    ) -> List[Dict]:
        """Merge headings on same line"""
        if not headings:
            return []
        
        sorted_headings = sorted(headings, key=lambda x: (x['page'], x['y_from_top'], x['x0']))
        
        merged = []
        current = None
        
        for h in sorted_headings:
            if current is None:
                current = h.copy()
                continue
            
            # Check merge conditions
            same_page = h['page'] == current['page']
            y_diff = abs(h['y_from_top'] - current['y_from_top'])
            same_line = same_page and y_diff <= y_tolerance
            
            same_font = (
                h['font_name'] == current['font_name'] and
                abs(h['font_size'] - current['font_size']) < 0.5 and
                h['is_bold'] == current['is_bold']
            )
            
            x_gap = h['x0'] - current['x1']
            close_enough = 0 <= x_gap <= x_gap_max
            
            if same_line and same_font and close_enough:
                # Merge
                current['text'] += ' ' + h['text']
                current['x1'] = h['x1']
                current['width'] = current['x1'] - current['x0']
                
                if h['is_all_caps']:
                    current['is_all_caps'] = True
                
                current['level'] = min(current['level'], h['level'])
            else:
                merged.append(current)
                current = h.copy()
        
        if current is not None:
            merged.append(current)
        
        return merged
    
    def _detect_table_regions(self, pdf_path: str) -> dict:
        """
        Detect table regions in PDF using pdfplumber
        
        Returns:
            Dict mapping page_num -> list of table bboxes (x0, y0, x1, y1)
        """
        try:
            import pdfplumber
        except ImportError:
            return {}
        
        table_regions = {}
        total_tables = 0
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.find_tables()
                    if tables:
                        total_tables += len(tables)
                        table_regions[page_num] = [
                            (table.bbox[0], table.bbox[1], table.bbox[2], table.bbox[3])
                            for table in tables
                        ]
            
            print(f"[DEBUG] Detected {total_tables} tables across {len(table_regions)} pages")
            
        except Exception as e:
            print(f"Warning: Table detection failed: {e}")
            return {}
        
        return table_regions
    
    def _is_in_table_region(self, page_num: int, element, table_regions: dict) -> bool:
        """
        Check if element overlaps with any table region on this page
        
        Args:
            page_num: Page number
            element: Text element with x0, y0, x1, y1 attributes
            table_regions: Dict from _detect_table_regions
            
        Returns:
            True if element is inside a table region
        """
        if page_num not in table_regions:
            return False
        
        elem_x0, elem_y0, elem_x1, elem_y1 = element.x0, element.y0, element.x1, element.y1
        
        for table_bbox in table_regions[page_num]:
            table_x0, table_y0, table_x1, table_y1 = table_bbox
            
            # Check if element overlaps with table bbox
            if (elem_x0 < table_x1 and elem_x1 > table_x0 and
                elem_y0 < table_y1 and elem_y1 > table_y0):
                return True
        
        return False
    
    def _is_non_heading_content(self, text: str) -> bool:
        """
        Check if text is obviously NOT a heading
        
        Filters out:
        - Dates (MM/DD/YYYY, etc.)
        - Times (HH:MM AM/PM)
        - URLs (www., .com, http)
        - Email addresses (@)
        - Page numbers (Page X of Y)
        - Pure numbers or decimal values
        - Parenthetical numbers like (0.250)
        - Phone numbers
        - Short standalone numbers
        
        Returns:
            True if text should be excluded
        """
        import re
        
        # Pattern definitions
        patterns = [
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Date: 01/16/2026
            r'\d{1,2}:\d{2}:\d{2}\s*(AM|PM|am|pm)',  # Time: 9:00:00 AM
            r'www\.|\.com|\.org|\.net|http',  # URL
            r'@',  # Email
            r'^Page\s+\d+\s+of\s+\d+',  # Page X of Y
            r'^\d{1,3}-\d{3,4}-\d{4}',  # Phone: 866-777-3638
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Pure numbers (integers or decimals)
        stripped = text.replace(',', '').replace('.', '').replace(' ', '')
        if stripped.isdigit():
            return True
        
        # Character composition analysis: filter if >70% is digits/symbols
        total_chars = len(text)
        digit_symbol_chars = sum(c.isdigit() or c in '.,()%$+-/' for c in text)
        
        if total_chars > 0:
            digit_ratio = digit_symbol_chars / total_chars
            if digit_ratio > 0.7:
                return True
        
        # Short text (<10 chars) with numbers (likely data labels or values)
        if len(text) < 10:
            has_digit = any(c.isdigit() for c in text)
            # If >50% is digits in short text, filter it
            if has_digit and digit_symbol_chars / total_chars > 0.5:
                return True
        
        # Single word that's very short (<=4 chars) - likely table header
        if len(text.split()) == 1 and len(text) <= 4 and text.isalpha():
            return True
        
        return False
