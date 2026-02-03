"""
Mortgage Document Layer Extraction Example using CompressionExtractor

This example demonstrates how to use CompressionExtractor to extract
Quick-Check and Shortlist layers from mortgage product documents.

Layers:
- Quick-Check: Hard eligibility rules (1-2k tokens)
- Shortlist: Compressed product features (8-10k tokens)

Requirements:
    - zag installed with CompressionExtractor
    - chak-ai installed: pip install chak-ai
    - OpenAI API key in .env file: OPENAI_API_KEY=xxx
"""

import os
import tempfile
import uuid
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel

# Load environment variables
load_dotenv()

from zag.extractors import CompressionExtractor
from zag.readers.mineru import MinerUReader


# Quick-Check extraction prompt
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


# Shortlist compression prompt
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


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("[bold cyan]Mortgage Layer Extraction with CompressionExtractor[/bold cyan]")
    print("=" * 80 + "\n")
    
    # Interactive file input
    print("Enter PDF file path (drag & drop supported): ", end="")
    try:
        path_input = input().strip()
    except KeyboardInterrupt:
        print("\n\n[yellow]Cancelled by user[/yellow]")
        return
    
    if not path_input:
        print("[red]No file path provided[/red]")
        return
    
    # Normalize path (handle PowerShell drag & drop)
    path_str = path_input.strip()
    if path_str.startswith("& "):
        path_str = path_str[2:].strip()
    elif path_str.startswith("&"):
        path_str = path_str[1:].strip()
    if (path_str.startswith("'") and path_str.endswith("'")) or (path_str.startswith('"') and path_str.endswith('"')):
        path_str = path_str[1:-1]
    pdf_path = Path(path_str)
    
    # Validate file
    if not pdf_path.exists():
        print(f"[red]Error: File not found: {pdf_path}[/red]")
        return
    
    # Step 1: Read PDF
    print("\n[yellow]📖 Step 1: Reading PDF...[/yellow]")
    reader = MinerUReader(backend="pipeline")
    doc = reader.read(str(pdf_path))
    
    extractor_instance = CompressionExtractor("openai/gpt-4o-mini")
    original_tokens = extractor_instance.count_tokens(doc.content)
    
    print(f"[green]✓ PDF read successfully[/green]")
    print(f"  Content: {len(doc.content):,} chars")
    print(f"  Tokens: {original_tokens:,}")
    print(f"  Pages: {len(doc.pages)}")
    
    # Configuration
    quick_check_target = 2000  # Fixed: hard rules layer size is relatively stable
    
    # Shortlist: maximize detail retention within LLM context window
    # Goal: keep as much detail as possible while fitting in one conversation
    EFFECTIVE_CONTEXT = 60000  # Model's effective context window
    shortlist_target = min(original_tokens, EFFECTIVE_CONTEXT)
    
    print("\n")
    print(Panel.fit(
        f"[bold green]Configuration[/bold green]\n" +
        f"PDF: {pdf_path.name}\n" +
        f"Quick-Check target: {quick_check_target} tokens\n" +
        f"Shortlist target: {shortlist_target} tokens\n" +
        f"LLM: OpenAI GPT-4o-mini",
        title="⚙️ Settings",
        border_style="blue"
    ))
    
    # Step 2: Extract Quick-Check layer
    print("\n[yellow]🔍 Step 2: Extracting Quick-Check layer...[/yellow]")
    quick_check_content = extractor_instance.compress(
        text=doc.content,
        prompt=QUICK_CHECK_PROMPT,
        target_tokens=quick_check_target,
        chunk_size=3000,
        max_depth=2
    )
    quick_check_tokens = extractor_instance.count_tokens(quick_check_content)
    
    print(f"[green]✓ Quick-Check extracted[/green]")
    print(f"  Tokens: {quick_check_tokens:,} ({quick_check_tokens/original_tokens:.1%})")
    
    # Step 3: Extract Shortlist layer
    print("\n[yellow]📋 Step 3: Extracting Shortlist layer...[/yellow]")
    shortlist_content = extractor_instance.compress(
        text=doc.content,
        prompt=SHORTLIST_PROMPT,
        target_tokens=shortlist_target,
        chunk_size=3000,
        max_depth=2
    )
    shortlist_tokens = extractor_instance.count_tokens(shortlist_content)
    
    print(f"[green]✓ Shortlist extracted[/green]")
    print(f"  Tokens: {shortlist_tokens:,} ({shortlist_tokens/original_tokens:.1%})")
    
    # Step 4: Save results to temp directory
    print("\n[yellow]💾 Step 4: Saving results...[/yellow]")
    
    # Create temp directory with random name
    temp_base = Path(tempfile.gettempdir())
    output_dir = temp_base / f"zag_mortgage_extraction_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = pdf_path.stem
    
    # Save original
    original_file = output_dir / f"{base_name}_original.md"
    with original_file.open("w", encoding="utf-8") as f:
        f.write(f"# Original Document: {pdf_path.name}\n\n")
        f.write(f"**Token count**: {original_tokens:,}\n\n")
        f.write("---\n\n")
        f.write(doc.content)
    
    # Save Quick-Check
    quick_check_file = output_dir / f"{base_name}_quick_check.md"
    with quick_check_file.open("w", encoding="utf-8") as f:
        f.write(quick_check_content)
    
    # Save Shortlist
    shortlist_file = output_dir / f"{base_name}_shortlist.md"
    with shortlist_file.open("w", encoding="utf-8") as f:
        f.write(shortlist_content)
    
    print(f"[green]✓ Files saved to:[/green]")
    print(f"  {output_dir}")
    print(f"\n  • {original_file.name}")
    print(f"  • {quick_check_file.name}")
    print(f"  • {shortlist_file.name}")
    
    # Final summary
    print("\n")
    print(Panel.fit(
        f"[bold green]✅ Extraction Complete![/bold green]\n\n" +
        f"Original: {original_tokens:,} tokens\n" +
        f"Quick-Check: {quick_check_tokens:,} tokens ({quick_check_tokens/original_tokens:.1%})\n" +
        f"Shortlist: {shortlist_tokens:,} tokens ({shortlist_tokens/original_tokens:.1%})\n\n" +
        f"📁 Output: {output_dir}",
        title="📊 Summary",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
