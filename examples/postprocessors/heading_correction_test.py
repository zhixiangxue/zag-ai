"""
Test heading correction with MinerUReader

This example demonstrates:
1. Reading PDF with MinerUReader
2. Extracting headings with HeadingExtractor
3. Correcting heading formats with HeadingCorrector
4. Outputting corrected doc.content and page.content to files
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from zag.readers.mineru import MinerUReader
from zag.postprocessors.correctors import HeadingCorrector

# Load environment variables
load_dotenv()


async def main():
    # Configuration
    pdf_path = input("Enter PDF path: ").strip().strip('"')
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Ask if user wants LLM correction
    use_llm = input("Use LLM correction? (y/n, default: y): ").strip().lower()
    use_llm = use_llm != 'n'  # Default to yes
    
    llm_uri = None
    api_key = None
    
    if use_llm:
        llm_uri = input("Enter LLM URI (default: openai/gpt-4o): ").strip()
        if not llm_uri:
            llm_uri = "openai/gpt-4o"
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Enter API key: ").strip()
            if not api_key:
                print("Error: API key is required for LLM correction")
                return
    
    # Output directory
    output_dir = Path("tmp")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = Path(pdf_path).stem
    
    print(f"\n{'='*60}")
    print(f"Testing Heading Correction")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path}")
    print(f"LLM Correction: {use_llm}")
    if use_llm:
        print(f"LLM: {llm_uri}")
    print(f"{'='*60}\n")
    
    # Step 1: Read PDF with MinerUReader
    print("Step 1: Reading PDF with MinerUReader...")
    reader = MinerUReader()
    pdf = reader.read(pdf_path)
    print(f"✓ Read {len(pdf.pages)} pages")
    
    # Save original content for comparison
    original_doc_file = output_dir / f"{pdf_name}_original_doc_{timestamp}.md"
    with open(original_doc_file, 'w', encoding='utf-8') as f:
        f.write(pdf.content)
    print(f"✓ Saved original doc.content to: {original_doc_file}")
    
    # Step 2: Extract raw headings first
    print("\nStep 2: Extracting raw headings...")
    from zag.extractors.heading import HeadingExtractor
    
    extractor = HeadingExtractor(
        llm_uri=llm_uri,
        api_key=api_key,
        llm_correction=use_llm
    )
    
    if use_llm:
        raw_headings = await extractor.aextract_from_pdf(pdf.metadata.source)
        print(f"✓ Extracted {len(raw_headings)} headings (with LLM)")
    else:
        raw_headings = extractor.extract_from_pdf(pdf.metadata.source)
        print(f"✓ Extracted {len(raw_headings)} headings (without LLM)")
    
    # Save raw headings to JSON for debugging
    import json
    raw_headings_file = output_dir / f"{pdf_name}_raw_headings_{timestamp}.json"
    with open(raw_headings_file, 'w', encoding='utf-8') as f:
        json.dump(raw_headings, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved raw headings to: {raw_headings_file}")
    
    # Step 3: Correct headings
    print("\nStep 3: Correcting heading formats...")
    corrector = HeadingCorrector(
        llm_uri=llm_uri,
        api_key=api_key,
        llm_correction=use_llm
    )
    
    if use_llm:
        pdf = await corrector.acorrect_document(pdf)
        print("✓ Heading correction completed (with LLM)")
    else:
        pdf = corrector.correct_document(pdf)
        print("✓ Heading correction completed (without LLM)")
    
    # Step 4: Save corrected content
    print("\nStep 4: Saving corrected content...")
    
    # Save corrected doc.content
    corrected_doc_file = output_dir / f"{pdf_name}_corrected_doc_{timestamp}.md"
    with open(corrected_doc_file, 'w', encoding='utf-8') as f:
        f.write(pdf.content)
    print(f"✓ Saved corrected doc.content to: {corrected_doc_file}")
    
    # Save each page's content
    pages_dir = output_dir / f"{pdf_name}_pages_{timestamp}"
    pages_dir.mkdir(exist_ok=True)
    
    for page in pdf.pages:
        page_file = pages_dir / f"page_{page.page_number:03d}.md"
        with open(page_file, 'w', encoding='utf-8') as f:
            f.write(f"# Page {page.page_number}\n\n")
            f.write(page.content)
    
    print(f"✓ Saved {len(pdf.pages)} page contents to: {pages_dir}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Raw headings (JSON): {raw_headings_file}")
    print(f"Original doc.content: {original_doc_file}")
    print(f"Corrected doc.content: {corrected_doc_file}")
    print(f"Page contents: {pages_dir}")
    print(f"{'='*60}\n")
    
    print("✓ Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
