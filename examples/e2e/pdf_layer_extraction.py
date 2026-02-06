"""End-to-end example: Extract multi-layer representations from mortgage PDF documents.

This script demonstrates a complete pipeline that:

    1. Reads a PDF file using MinerU
    2. Extracts Quick-Check layer (hard eligibility rules, ~2k tokens)
    3. Extracts Shortlist layer (compressed features, ~60k tokens)
    4. Converts to Markdown and builds DocTree structure
    5. Returns all four layers: t1(quick_check), t2(shortlist), t3(tree), t4(original)

Layers:
    - t1 (Quick-Check): Hard eligibility rules for fast screening
    - t2 (Shortlist): Compressed product features for detailed analysis
    - t3 (Tree): Hierarchical DocTree structure for retrieval
    - t4 (Original): Full document content

Dependencies (install as needed):
    pip install rich diskcache xxhash tiktoken python-dotenv

Run:
    python examples/e2e/pdf_layer_extraction.py
"""

import os
import tempfile
import uuid
import json
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from zag.readers import MinerUReader, MarkdownTreeReader
from zag.extractors import CompressionExtractor
from zag.schemas import DocTree, BaseUnit, ContentView, LODLevel

# Load environment variables
load_dotenv()

console = Console()

# Get API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    console.print("[bold red]Warning: OPENAI_API_KEY not found in environment. Please set it in .env file.[/bold red]")


# ---------------------------------------------------------------------------
# Prompts (copied from compression_mortgage_example.py)
# ---------------------------------------------------------------------------

QUICK_CHECK_PROMPT = """You are a mortgage document analysis expert. Extract the Quick-Check layer from the following document.

**What is the Quick-Check layer?**
This layer contains hard "one-strike-out" eligibility criteria used to quickly eliminate non-qualifying applications. Loan officers use this layer first to determine if an applicant meets basic qualifications.

**Typical characteristics**:
- Uses negation words like "NOT ELIGIBLE", "ineligible", "prohibited", "not permitted"
- Involves minimum/maximum numeric limits (credit score, LTV, loan amount)
- Clear restrictions on borrower identity, property type, geographic location
- Simple conditions that can be quickly evaluated (no complex calculations)
- Usually found at the beginning of documents (Eligibility, Requirements, Restrictions sections)

**Extraction requirements**:
1. Extract only hard rejection criteria (not "recommended" or "preferred")
2. Preserve complete logical relationships (AND/OR/IF-THEN/EXCEPT)
3. Preserve all numbers, percentages, entity names
4. Use concise but complete natural language
5. Maintain original section structure
6. Target length: 1-2k tokens

**Few-shot examples**:

Example input 1:
"Borrowers with an Individual Taxpayer Identification Number (ITIN) are not eligible. All borrowers must have a valid social security number."

Extraction output:
"### Borrower Type
- ITIN holders: NOT ELIGIBLE
- Must have valid social security number"

---

Example input 2:
"Minimum Credit Score: 700 (for DSCR ≥ 1.00) or 720 (for DSCR 0.80 - 1.00)"

Extraction output:
"### Credit Score
- Minimum credit score: 700 (when DSCR ≥ 1.00)
- Minimum credit score: 720 (when DSCR 0.80 - 1.00)"

---

**Now analyze the following document and extract the Quick-Check layer**:

{text}

**Output requirements**:
- Maintain Markdown format
- Organize content with section headings
- Each criterion on a separate line
- Include only hard rejection criteria
"""


SHORTLIST_PROMPT = """You are a professional mortgage document compression expert. 

**CRITICAL REQUIREMENT**: You MUST compress the following text to EXACTLY {target_tokens} tokens or less. The current text has too many tokens and MUST be reduced.

**Absolutely preserve (NEVER delete or modify)**:
1. All numbers, percentages, amounts, dates (e.g., 700 score, 80% LTV, $125,000, 6 months)
2. All placeholders in format {{{{HTML_TABLE_X}}}} (e.g., {{{{HTML_TABLE_0}}}}, {{{{HTML_TABLE_1}}}}) - these are table references and MUST remain unchanged
3. All eligibility condition if/then logic relationships
4. All entity names (company names, product names, location names, state names)
5. All negation expressions and restriction clauses (NOT ELIGIBLE, prohibited, not permitted, ineligible)

**What to compress (actively remove)**:
- Verbose explanations and redundant descriptions
- Repetitive legal statements and disclaimers
- Multiple similar examples (keep only 1 representative example)
- Overly detailed process descriptions
- Explanatory text outside of tables

**Compression priorities**:
- Priority 1: Preserve core business rules and ALL numbers (most important)
- Priority 2: Maintain logical relationship integrity (if/and/or/but/except conditions)
- Priority 3: Actively remove verbose language and redundant content
- Priority 4: Remove all unnecessary formatting and whitespace

**Output requirements**:
- MUST be {target_tokens} tokens or LESS
- Do NOT add any summary language (e.g., "In summary", "In conclusion")
- Output compressed content directly without extra explanations
- Maintain Markdown format but remove unnecessary formatting
- Keep all {{{{HTML_TABLE_X}}}} placeholders exactly as they appear

Original text:
{text}

Output compressed text (MUST be <= {target_tokens} tokens):"""


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process_pdf_document(
    pdf_path: str,
    quick_check_target: int = 2000,
    shortlist_target: int = 60000,
    llm_uri: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    save_to_temp: bool = True
) -> Dict[str, any]:
    """Extract multi-layer representations from a PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        quick_check_target: Target token count for Quick-Check layer (default: 2000)
        shortlist_target: Target token count for Shortlist layer (default: 60000)
        llm_uri: LLM URI for extraction and tree generation
        api_key: API key for LLM (defaults to OPENAI_API_KEY from env)
        save_to_temp: Whether to save outputs to temporary directory
    
    Returns:
        Dict containing:
            - unit: BaseUnit with views containing all 4 layers
            - temp_dir: Optional[Path] (Temporary directory path if saved)
    """
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]PDF Multi-Layer Extraction Pipeline[/bold cyan]")
    console.print("=" * 80 + "\n")
    
    # Step 1: Read PDF with MinerU
    console.print("[yellow]Step 1: Reading PDF with MinerU...[/yellow]")
    
    reader = MinerUReader(backend="pipeline")
    doc = reader.read(str(pdf_path))
    lod_full = doc.content
    
    extractor = CompressionExtractor(llm_uri)
    original_tokens = extractor.count_tokens(lod_full)
    
    console.print(f"[green]✓ PDF read successfully[/green]")
    console.print(f"  Content: {len(lod_full):,} chars")
    console.print(f"  Tokens: {original_tokens:,}")
    console.print(f"  Pages: {len(doc.pages)}")
    
    # Step 2: Extract Quick-Check layer (lod_low)
    console.print("\n[yellow]Step 2: Extracting Quick-Check layer (lod_low)...[/yellow]")
    
    lod_low = extractor.compress(
        text=lod_full,
        prompt=QUICK_CHECK_PROMPT,
        target_tokens=quick_check_target,
        chunk_size=3000,
        max_depth=2
    )
    lod_low_tokens = extractor.count_tokens(lod_low)
    
    console.print(f"[green]✓ Quick-Check extracted[/green]")
    console.print(f"  Tokens: {lod_low_tokens:,} ({lod_low_tokens/original_tokens:.1%})")
    
    # Step 3: Extract Shortlist layer (lod_medium)
    console.print("\n[yellow]Step 3: Extracting Shortlist layer (lod_medium)...[/yellow]")
    
    lod_medium = extractor.compress(
        text=lod_full,
        prompt=SHORTLIST_PROMPT,
        target_tokens=shortlist_target,
        chunk_size=3000,
        max_depth=2
    )
    lod_medium_tokens = extractor.count_tokens(lod_medium)
    
    console.print(f"[green]✓ Shortlist extracted[/green]")
    console.print(f"  Tokens: {lod_medium_tokens:,} ({lod_medium_tokens/original_tokens:.1%})")
    
    # Step 4: Generate DocTree from Markdown (lod_high)
    console.print("\n[yellow]Step 4: Building DocTree structure (lod_high)...[/yellow]")
    
    try:
        tree_reader = MarkdownTreeReader(llm_uri=llm_uri, api_key=api_key)
        lod_tree = tree_reader.read(content=lod_full, generate_summaries=True)
        
        all_nodes = lod_tree.collect_all_nodes()
        console.print(f"[green]✓ DocTree built successfully[/green]")
        console.print(f"  Total nodes: {len(all_nodes)}")
        console.print(f"  Root: {lod_tree.doc_name}")
    except Exception as e:
        console.print(f"[red]✗ DocTree building failed: {e}[/red]")
        raise
    
    # Step 5: Save to temporary directory
    temp_dir = None
    if save_to_temp:
        console.print("\n[yellow]Step 5: Saving outputs to temporary directory...[/yellow]")
        
        temp_base = Path(tempfile.gettempdir())
        temp_dir = temp_base / f"zag_pdf_extraction_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = pdf_path.stem
        
        # Save all layers
        (temp_dir / f"{base_name}_lod_low.md").write_text(lod_low, encoding="utf-8")
        (temp_dir / f"{base_name}_lod_medium.md").write_text(lod_medium, encoding="utf-8")
        (temp_dir / f"{base_name}_lod_full.md").write_text(lod_full, encoding="utf-8")
        
        # Save lod_tree as JSON (can be loaded back with DocTree.from_json)
        lod_tree.to_json(str(temp_dir / f"{base_name}_lod_tree.json"))
        
        console.print(f"[green]✓ Files saved to:[/green]")
        console.print(f"  {temp_dir}")
        console.print(f"\n  Files:")
        console.print(f"    • {base_name}_lod_low.md")
        console.print(f"    • {base_name}_lod_medium.md")
        console.print(f"    • {base_name}_lod_tree.json")
        console.print(f"    • {base_name}_lod_full.md")
    
    # Return BaseUnit with views
    unit = BaseUnit(
        unit_id=f"pdf_{uuid.uuid4().hex[:12]}",
        content=lod_low,  # Primary content for retrieval
        embedding_content=lod_low,  # Full content for embedding
        views=[
            ContentView(level=LODLevel.LOW, content=lod_low, token_count=lod_low_tokens),
            ContentView(level=LODLevel.MEDIUM, content=lod_medium, token_count=lod_medium_tokens),
            ContentView(level=LODLevel.HIGH, content=lod_tree.to_dict()),
            ContentView(level=LODLevel.FULL, content=lod_full, token_count=original_tokens)
        ]
    )
    
    # Print summary
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]Extraction Complete[/bold green]\n\n" +
        f"lod_full (Original): {original_tokens:,} tokens\n" +
        f"lod_low (Quick-Check): {lod_low_tokens:,} tokens ({lod_low_tokens/original_tokens:.1%})\n" +
        f"lod_medium (Shortlist): {lod_medium_tokens:,} tokens ({lod_medium_tokens/original_tokens:.1%})\n" +
        f"lod_tree (Tree): {len(lod_tree.collect_all_nodes())} nodes\n\n" +
        (f"Output directory: {temp_dir}" if temp_dir else "No files saved"),
        title="Summary",
        border_style="green"
    ))
    
    return {
        "unit": unit,
        "temp_dir": temp_dir
    }


# ---------------------------------------------------------------------------
# UI helper
# ---------------------------------------------------------------------------

def _clean_path(input_path: str) -> str:
    """Clean Windows drag-and-drop path quirks."""
    path = input_path.strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if path.startswith("'") and path.endswith("'"):
        path = path[1:-1]
    if path.startswith("& "):
        path = path[2:].strip()
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
    return path


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    """Interactive CLI for PDF layer extraction."""
    console.print("\n[bold cyan]PDF Multi-Layer Extraction Tool[/bold cyan]\n")
    
    # Interactive file input
    console.print("Enter PDF file path (drag & drop supported): ", end="")
    try:
        path_input = input().strip()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Cancelled by user[/yellow]")
        return
    
    if not path_input:
        console.print("[red]No file path provided[/red]")
        return
    
    pdf_path = Path(_clean_path(path_input))
    
    if not pdf_path.exists():
        console.print(f"[red]Error: File not found: {pdf_path}[/red]")
        return
    
    if not pdf_path.suffix.lower() == ".pdf":
        console.print(f"[red]Error: Not a PDF file: {pdf_path}[/red]")
        return
    
    # Process document
    try:
        result = process_pdf_document(
            pdf_path=str(pdf_path),
            save_to_temp=True
        )
                
        unit = result["unit"]
        console.print("\n[bold green]Processing complete![/bold green]")
        console.print(f"\nCreated Unit with {len(unit.views)} views:")
        for view in unit.get_all_views():
            console.print(f"  - {view.level.value}: {view.token_count or 'N/A'} tokens")
                
        # Example usage
        console.print("\nExample usage:")
        console.print(f"  quick_check = unit.get_view(LODLevel.LOW)")
        console.print(f"  shortlist = unit.get_view(LODLevel.MEDIUM)")
        console.print(f"  tree_dict = unit.get_view(LODLevel.HIGH)")
        console.print(f"  original = unit.get_view(LODLevel.FULL)")
    
    except Exception as e:
        console.print(f"\n[bold red]Error during processing:[/bold red] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
