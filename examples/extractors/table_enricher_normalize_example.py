"""
TableEnricher with normalization example

Test table structure normalization for complex tables with:
- Multi-level headers
- Merged cells
- Empty cells
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from zag.readers.camelot import CamelotReader
from zag.extractors.table_enricher import TableEnricher


async def main():
    # Load environment variables
    load_dotenv()
    
    # Get PDF file path from user
    pdf_path_input = input("Enter PDF file path: ").strip().strip('"').strip("'")
    pdf_path = Path(pdf_path_input)
    
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return
    
    print(f"\n📄 Reading PDF: {pdf_path}")
    
    # Stage 1: Extract tables using CamelotReader
    print("\n🔍 Stage 1: Extracting tables with Camelot...")
    reader = CamelotReader(flavor="stream")
    pdf_doc = reader.read(str(pdf_path))
    table_units = list(pdf_doc.units)
    
    print(f"✅ Extracted {len(table_units)} tables")
    
    if not table_units:
        print("⚠️  No tables found in PDF")
        return
    
    # Show original tables
    print("\n📊 Original tables:")
    for i, table_unit in enumerate(table_units, 1):
        print(f"\n  Table {i}:")
        print(f"    Shape: {table_unit.df.shape}")
        print(f"    Columns: {table_unit.df.columns.tolist()[:5]}...")
        print(f"    Content preview (first 3 rows):")
        print(f"    {table_unit.df.head(3).to_string(max_colwidth=20)}")
    
    # Stage 2: Enrich with normalization
    print("\n🔧 Stage 2: Enriching with normalization enabled...")
    
    # Get LLM config from environment
    llm_uri = os.getenv("LLM_URI", "openai/gpt-4o")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set, using default")
    
    enricher = TableEnricher(
        llm_uri=llm_uri,
        api_key=api_key,
        normalize_table=True  # Enable normalization
    )
    
    await enricher.aextract(table_units)
    
    print(f"✅ Enrichment complete")
    
    # Show normalized tables
    print("\n📊 Normalized tables:")
    for i, table_unit in enumerate(table_units, 1):
        print(f"\n  Table {i}:")
        print(f"    Shape: {table_unit.df.shape}")
        print(f"    Columns: {table_unit.df.columns.tolist()[:5]}...")
        print(f"    Caption: {table_unit.caption}")
        print(f"    Content preview (first 3 rows):")
        print(f"    {table_unit.df.head(3).to_string(max_colwidth=20)}")
        
        # Save to file for inspection
        output_dir = Path("playground/table_enricher_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original
        original_file = output_dir / f"table_{i}_original.md"
        with open(original_file, 'w', encoding='utf-8') as f:
            # Note: We can't get original content anymore, just show df
            f.write(f"# Table {i} - Original\n\n")
            f.write(f"Shape: {table_unit.df.shape}\n\n")
            f.write(f"Columns: {table_unit.df.columns.tolist()}\n\n")
            f.write("## Content\n\n")
            f.write(table_unit.content)
        
        # Save normalized
        normalized_file = output_dir / f"table_{i}_normalized.md"
        with open(normalized_file, 'w', encoding='utf-8') as f:
            f.write(f"# Table {i} - Normalized\n\n")
            f.write(f"Shape: {table_unit.df.shape}\n\n")
            f.write(f"Columns: {table_unit.df.columns.tolist()}\n\n")
            f.write(f"Caption: {table_unit.caption}\n\n")
            f.write("## Content\n\n")
            f.write(table_unit.content)
        
        print(f"    💾 Saved: {original_file}")
        print(f"    💾 Saved: {normalized_file}")


if __name__ == "__main__":
    asyncio.run(main())
