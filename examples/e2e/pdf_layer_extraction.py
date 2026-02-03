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
from zag.schemas import DocTree

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
    save_to_temp: bool = True,
    verbose: bool = True
) -> Dict[str, any]:
    """Extract multi-layer representations from a PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        quick_check_target: Target token count for Quick-Check layer (default: 2000)
        shortlist_target: Target token count for Shortlist layer (default: 60000)
        llm_uri: LLM URI for extraction and tree generation
        api_key: API key for LLM (defaults to OPENAI_API_KEY from env)
        save_to_temp: Whether to save outputs to temporary directory
        verbose: Whether to print progress information
    
    Returns:
        Dict containing:
            - t1_quick_check: str (Quick-Check layer content)
            - t2_shortlist: str (Shortlist layer content)
            - t3_tree: DocTree (Hierarchical tree structure)
            - t4_original: str (Original document content)
            - temp_dir: Optional[Path] (Temporary directory path if saved)
    """
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if verbose:
        console.print("\n" + "=" * 80)
        console.print("[bold cyan]PDF Multi-Layer Extraction Pipeline[/bold cyan]")
        console.print("=" * 80 + "\n")
    
    # Step 1: Read PDF with MinerU
    if verbose:
        console.print("[yellow]Step 1: Reading PDF with MinerU...[/yellow]")
    
    reader = MinerUReader(backend="pipeline")
    doc = reader.read(str(pdf_path))
    t4_original = doc.content
    
    extractor = CompressionExtractor(llm_uri)
    original_tokens = extractor.count_tokens(t4_original)
    
    if verbose:
        console.print(f"[green]✓ PDF read successfully[/green]")
        console.print(f"  Content: {len(t4_original):,} chars")
        console.print(f"  Tokens: {original_tokens:,}")
        console.print(f"  Pages: {len(doc.pages)}")
    
    # Step 2: Extract Quick-Check layer (t1)
    if verbose:
        console.print("\n[yellow]Step 2: Extracting Quick-Check layer (t1)...[/yellow]")
    
    t1_quick_check = extractor.compress(
        text=t4_original,
        prompt=QUICK_CHECK_PROMPT,
        target_tokens=quick_check_target,
        chunk_size=3000,
        max_depth=2
    )
    t1_tokens = extractor.count_tokens(t1_quick_check)
    
    if verbose:
        console.print(f"[green]✓ Quick-Check extracted[/green]")
        console.print(f"  Tokens: {t1_tokens:,} ({t1_tokens/original_tokens:.1%})")
    
    # Step 3: Extract Shortlist layer (t2)
    if verbose:
        console.print("\n[yellow]Step 3: Extracting Shortlist layer (t2)...[/yellow]")
    
    t2_shortlist = extractor.compress(
        text=t4_original,
        prompt=SHORTLIST_PROMPT,
        target_tokens=shortlist_target,
        chunk_size=3000,
        max_depth=2
    )
    t2_tokens = extractor.count_tokens(t2_shortlist)
    
    if verbose:
        console.print(f"[green]✓ Shortlist extracted[/green]")
        console.print(f"  Tokens: {t2_tokens:,} ({t2_tokens/original_tokens:.1%})")
    
    # Step 4: Generate DocTree from Markdown (t3)
    if verbose:
        console.print("\n[yellow]Step 4: Building DocTree structure (t3)...[/yellow]")
    
    # Save markdown to temporary file for MarkdownTreeReader
    temp_md_path = Path(tempfile.gettempdir()) / f"zag_temp_{uuid.uuid4().hex[:8]}.md"
    with temp_md_path.open("w", encoding="utf-8") as f:
        f.write(t4_original)
    
    try:
        tree_reader = MarkdownTreeReader(llm_uri=llm_uri, api_key=api_key)
        t3_tree = tree_reader.read(path=str(temp_md_path), generate_summaries=True)
        
        if verbose:
            all_nodes = t3_tree.collect_all_nodes()
            console.print(f"[green]✓ DocTree built successfully[/green]")
            console.print(f"  Total nodes: {len(all_nodes)}")
            console.print(f"  Root: {t3_tree.doc_name}")
    finally:
        # Clean up temporary markdown file
        if temp_md_path.exists():
            temp_md_path.unlink()
    
    # Step 5: Save to temporary directory
    temp_dir = None
    if save_to_temp:
        if verbose:
            console.print("\n[yellow]Step 5: Saving outputs to temporary directory...[/yellow]")
        
        temp_base = Path(tempfile.gettempdir())
        temp_dir = temp_base / f"zag_pdf_extraction_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = pdf_path.stem
        
        # Save all layers
        (temp_dir / f"{base_name}_t1_quick_check.md").write_text(t1_quick_check, encoding="utf-8")
        (temp_dir / f"{base_name}_t2_shortlist.md").write_text(t2_shortlist, encoding="utf-8")
        (temp_dir / f"{base_name}_t4_original.md").write_text(t4_original, encoding="utf-8")
        
        # Save t3_tree as JSON (can be loaded back with DocTree.from_json)
        t3_tree.to_json(str(temp_dir / f"{base_name}_t3_tree.json"))
        
        if verbose:
            console.print(f"[green]✓ Files saved to:[/green]")
            console.print(f"  {temp_dir}")
            console.print(f"\n  Files:")
            console.print(f"    • {base_name}_t1_quick_check.md")
            console.print(f"    • {base_name}_t2_shortlist.md")
            console.print(f"    • {base_name}_t3_tree.json")
            console.print(f"    • {base_name}_t4_original.md")
    
    # Print summary
    if verbose:
        console.print("\n")
        console.print(Panel.fit(
            f"[bold green]Extraction Complete[/bold green]\n\n" +
            f"t4 (Original): {original_tokens:,} tokens\n" +
            f"t1 (Quick-Check): {t1_tokens:,} tokens ({t1_tokens/original_tokens:.1%})\n" +
            f"t2 (Shortlist): {t2_tokens:,} tokens ({t2_tokens/original_tokens:.1%})\n" +
            f"t3 (Tree): {len(t3_tree.collect_all_nodes())} nodes\n\n" +
            (f"Output directory: {temp_dir}" if temp_dir else "No files saved"),
            title="Summary",
            border_style="green"
        ))
    
    return {
        "t1_quick_check": t1_quick_check,
        "t2_shortlist": t2_shortlist,
        "t3_tree": t3_tree,
        "t4_original": t4_original,
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
            save_to_temp=True,
            verbose=True
        )
        
        console.print("\n[bold green]Processing complete! Results ready for embedding and Qdrant storage.[/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error during processing:[/bold red] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
