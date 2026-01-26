#!/usr/bin/env python3
"""
Test TableParser and TableEnricher - Complete workflow

Demonstrates:
- TableParser: Extract tables from TextUnit and generate DataFrame
- TableEnricher: Generate caption, embedding_content, and data-critical detection
- Relations: Bidirectional linking between TextUnit and TableUnit
- Complete workflow from parsing to enrichment

Note: Uses OpenAI gpt-4o (requires OPENAI_API_KEY in .env)
"""

import sys
import asyncio
from pathlib import Path
from rich import print as rprint

from zag.parsers import TableParser
from zag.extractors import TableEnricher
from zag.schemas.unit import TextUnit
from zag.schemas import UnitMetadata


# Sample document with tables
SAMPLE_CONTENT = """# Mortgage Products Overview

We offer a variety of mortgage products to meet your needs.

## Fixed-Rate Mortgages

Fixed-rate mortgages provide stable monthly payments throughout the loan term.

| Product Type        | Term    | Interest Rate | APR    | Minimum Down Payment |
| ------------------- | ------- | ------------- | ------ | -------------------- |
| Fixed-Rate Mortgage | 30-Year | 6.125%        | 6.275% | 3%                   |
| Fixed-Rate Mortgage | 15-Year | 5.750%        | 5.920% | 5%                   |
| Fixed-Rate Mortgage | 20-Year | 5.875%        | 6.050% | 5%                   |

These rates are for borrowers with excellent credit scores (720+).

## Adjustable-Rate Mortgages (ARMs)

ARMs offer lower initial rates that adjust after a fixed period.

| Product Type | Term    | Initial Rate | APR    | Fixed Period |
| ------------ | ------- | ------------ | ------ | ------------ |
| 5/1 ARM      | 30-Year | 5.875%       | 6.125% | 5 years      |
| 7/1 ARM      | 30-Year | 6.000%       | 6.250% | 7 years      |
| 10/1 ARM     | 30-Year | 6.125%       | 6.375% | 10 years     |

ARM rates adjust annually after the fixed period based on market conditions.
"""


async def main():
    print("=" * 70)
    print("TableParser + TableEnricher Complete Workflow Test")
    print("=" * 70)

    # Step 1: Create TextUnit (simulating document splitting)
    print("\n" + "=" * 70)
    print("Step 1: Create TextUnit from document")
    print("=" * 70)

    text_unit = TextUnit(
        unit_id="text_001",
        content=SAMPLE_CONTENT,
        metadata=UnitMetadata(
            context_path="Mortgage Products/Overview",
            page_numbers=[1, 2]
        )
    )

    print(f"\n✅ Created TextUnit:")
    print(f"  Unit ID: {text_unit.unit_id}")
    print(f"  Content length: {len(text_unit.content)} chars")
    print(f"  Context path: {text_unit.metadata.context_path}")
    print(f"  Page numbers: {text_unit.metadata.page_numbers}")

    # Step 2: Parse tables from TextUnit
    print("\n" + "=" * 70)
    print("Step 2: TableParser extracts tables")
    print("=" * 70)

    parser = TableParser()
    table_units = parser.parse_from_unit(text_unit)

    print(f"\n✅ Parsed {len(table_units)} tables")

    for i, table in enumerate(table_units, 1):
        print(f"\nTable {i}:")
        print(f"  Unit ID: {table.unit_id}")
        print(f"  DataFrame shape: {table.df.shape}")
        print(f"  Columns: {list(table.df.columns)}")
        print(f"  Rows: {len(table.df)}")
        print(f"  Metadata inherited: {table.metadata.context_path}")
        print(f"  Page numbers: {table.metadata.page_numbers}")

        # Check relations
        referenced_by = table.get_referenced_by()
        print(f"  Referenced by: {len(referenced_by)} TextUnit(s)")
        if referenced_by:
            print(f"    - TextUnit ID: {referenced_by[0].unit_id}")

    # Verify bidirectional relations
    print("\n📊 Verify bidirectional relations:")
    references = text_unit.get_references()
    print(f"  TextUnit references {len(references)} TableUnit(s)")
    for ref in references:
        print(f"    - TableUnit ID: {ref.unit_id}")

    # Step 3: Enrich TableUnits
    print("\n" + "=" * 70)
    print("Step 3: TableEnricher generates enrichment data")
    print("=" * 70)

    # Check for API key
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  Warning: OPENAI_API_KEY not found in .env")
        print("   Skipping enrichment step (requires LLM)")
        print("\n💡 To test enrichment, add your API key to .env:")
        print("   OPENAI_API_KEY=sk-xxx")
        return

    print(f"\n✅ Using OpenAI gpt-4o")
    print(f"   API key: {api_key[:10]}...")

    enricher = TableEnricher(
        llm_uri="openai/gpt-4o",
        api_key=api_key
    )

    print("\n⏳ Generating enrichment data (may take 10-20 seconds)...")
    results = await enricher.aextract(table_units)

    print("\n✅ Enrichment complete!")

    # Note: TableEnricher automatically enriches units in-place:
    # - caption → unit.caption
    # - embedding_content → unit.embedding_content
    # - is_data_critical, criticality_reason → unit.metadata.custom

    # Step 4: Verify enriched TableUnits
    print("\n" + "=" * 70)
    print("Step 4: Verify enriched TableUnits")
    print("=" * 70)

    # Display enriched TableUnits
    for i, table in enumerate(table_units, 1):
        print(f"\n{'='*60}")
        print(f"Enriched Table {i}")
        print(f"{'='*60}")
        print(f"  Caption: {table.caption or 'N/A'}")
        
        # Note: is_data_critical is stored in metadata.custom["table"]
        table_meta = table.metadata.custom.get('table', {})
        print(f"  Is Data-Critical: {table_meta.get('is_data_critical', 'N/A')}")
        print(f"  Criticality Reason: {table_meta.get('criticality_reason', 'N/A')}")
        
        print(f"  Embedding Content Preview: {table.embedding_content[:150] if table.embedding_content else 'N/A'}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
