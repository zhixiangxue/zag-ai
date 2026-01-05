"""
Test MarkdownHeaderSplitter
"""

from zag.readers.markitdown import MarkItDownReader
from zag.splitters.markdown import MarkdownHeaderSplitter


def test_markdown_header_splitter():
    """Test splitting markdown by headers"""
    print("\n" + "="*60)
    print("Testing MarkdownHeaderSplitter")
    print("="*60)
    
    # Create test markdown content
    markdown_content = """# Introduction
This is the introduction section.
It has multiple lines.

## Background
This is the background section.

### History
This is the history subsection.

## Goals
These are the goals.

# Conclusion
This is the conclusion.
"""
    
    # Create markdown document
    from zag.schemas.markdown import Markdown
    from zag.schemas.base import DocumentMetadata
    
    doc = Markdown(
        content=markdown_content,
        metadata=DocumentMetadata(
            source="test.md",
            source_type="local",
            file_type="markdown",
            content_length=len(markdown_content)
        )
    )
    
    print(f"\n✓ Document created with ID: {doc.doc_id}")
    print(f"  Content length: {len(doc.content)} chars")
    
    # Split by headers
    splitter = MarkdownHeaderSplitter()
    units = doc.split(splitter)
    
    print(f"\n✓ Split into {len(units)} units")
    
    # Display each unit
    for i, unit in enumerate(units):
        print(f"\n--- Unit {i+1} ---")
        print(f"  ID: {unit.unit_id}")
        print(f"  Context Path: {unit.metadata.context_path}")
        print(f"  Content (first 50 chars): {unit.content[:50]}...")
        print(f"  Prev Unit ID: {unit.prev_unit_id}")
        print(f"  Next Unit ID: {unit.next_unit_id}")
        print(f"  Source Doc ID: {unit.source_doc_id}")
    
    # Verify
    assert len(units) == 5, f"Expected 5 units, got {len(units)}"
    assert units[0].metadata.context_path == "Introduction"
    assert units[1].metadata.context_path == "Introduction/Background"
    assert units[2].metadata.context_path == "Introduction/Background/History"
    assert units[3].metadata.context_path == "Introduction/Goals"
    assert units[4].metadata.context_path == "Conclusion"
    
    # Verify chain
    assert units[0].prev_unit_id is None
    assert units[0].next_unit_id == units[1].unit_id
    assert units[1].prev_unit_id == units[0].unit_id
    assert units[4].next_unit_id is None
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_markdown_header_splitter()
