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
    
    # Read markdown content from test file
    import os
    markdown_file_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'files',
        'mortgage_products.md'
    )
    
    print(f"\nReading file: {markdown_file_path}")
    with open(markdown_file_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    print(f"✓ File loaded: {len(markdown_content)} chars")
    
    # Create markdown document
    from zag.schemas.markdown import Markdown
    from zag.schemas import DocumentMetadata
    
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
    
    # Display each unit (first 10 only for readability)
    display_count = min(10, len(units))
    for i in range(display_count):
        unit = units[i]
        print(f"\n--- Unit {i+1} ---")
        print(f"  ID: {unit.unit_id}")
        print(f"  Context Path: {unit.metadata.context_path}")
        print(f"  Content length: {len(unit.content)} chars")
        print(f"  Content preview: {unit.content[:80].replace(chr(10), ' ')}...")
        print(f"  Prev Unit ID: {unit.prev_unit_id}")
        print(f"  Next Unit ID: {unit.next_unit_id}")
    
    if len(units) > display_count:
        print(f"\n... and {len(units) - display_count} more units")
    
    # Verify chain integrity
    print("\n✓ Verifying unit chain...")
    assert units[0].prev_unit_id is None, "First unit should have no previous"
    assert units[-1].next_unit_id is None, "Last unit should have no next"
    
    for i in range(len(units) - 1):
        assert units[i].next_unit_id == units[i+1].unit_id, f"Chain broken at unit {i}"
        assert units[i+1].prev_unit_id == units[i].unit_id, f"Chain broken at unit {i+1}"
    
    print("  ✓ All units properly chained")
    
    # Display context path hierarchy
    print("\n✓ Context Path Hierarchy:")
    for i, unit in enumerate(units[:10]):
        print(f"  {i+1}. {unit.metadata.context_path}")
    if len(units) > 10:
        print(f"  ... and {len(units) - 10} more")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_markdown_header_splitter()
