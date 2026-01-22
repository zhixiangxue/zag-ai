"""
Test MarkItDownReader.read() function
"""

from pathlib import Path
from zag.readers.markitdown import MarkItDownReader
from zag.schemas.pdf import PDF


def normalize_path(path_str: str) -> Path:
    """
    Normalize file path by removing quotes and handling drag & drop paths.
    
    Handles:
    - Single quotes: '/path/to/file.pdf'
    - Double quotes: "/path/to/file.pdf"
    - Escaped spaces: /path/to/my\\ file.pdf
    - Plain paths: /path/to/file.pdf
    """
    # Remove leading/trailing whitespace
    path_str = path_str.strip()
    
    # Remove surrounding quotes (single or double)
    if (path_str.startswith("'") and path_str.endswith("'")) or \
       (path_str.startswith('"') and path_str.endswith('"')):
        path_str = path_str[1:-1]
    
    # Handle escaped spaces (unescape them)
    path_str = path_str.replace("\\ ", " ")
    
    return Path(path_str)


def test_read_pdf(file_path: str):
    """Test reading PDF file"""
    print("\n" + "="*60)
    print("MarkItDown Reader Test")
    print("="*60)
    
    reader = MarkItDownReader()
    
    print(f"\nReading: {file_path}")
    doc = reader.read(file_path)
    
    # Save to temp directory
    import tempfile
    import os
    temp_dir = Path(tempfile.gettempdir())
    output_file = temp_dir / f"{Path(file_path).stem}_markitdown.md"
    
    with output_file.open('w', encoding='utf-8') as f:
        f.write(doc.content)
    
    print(f"\n✓ Content saved to: {output_file}")
    
    print(f"\n✓ Document created successfully")
    print(f"  - Type: {type(doc).__name__}")
    print(f"  - Doc ID: {doc.doc_id}")
    print(f"  - Is PDF: {isinstance(doc, PDF)}")
    print(f"\n✓ Metadata:")
    print(f"  - source: {doc.metadata.source}")
    print(f"  - source_type: {doc.metadata.source_type}")
    print(f"  - file_type: {doc.metadata.file_type}")
    print(f"  - file_name: {doc.metadata.file_name}")
    print(f"  - file_size: {doc.metadata.file_size} bytes")
    print(f"  - file_extension: {doc.metadata.file_extension}")
    print(f"  - content_length: {doc.metadata.content_length} characters")
    print(f"  - reader_name: {doc.metadata.reader_name}")
    print(f"  - created_at: {doc.metadata.created_at}")
    
    print(f"\n✓ Content preview (first 500 chars):")
    print(f"  {doc.content[:500]}...")
    
    return doc


def main():
    """Run tests"""
    print("\n" + "="*60)
    print("MARKITDOWN READER - INTERACTIVE TEST")
    print("="*60 + "\n")
    
    print("Enter PDF file path (drag & drop supported): ", end="")
    try:
        path_input = input().strip()
    except KeyboardInterrupt:
        print("\n\n[yellow]Cancelled by user[/yellow]")
        return
    
    if not path_input:
        print("No file path provided")
        return
    
    # Normalize path (handle quotes and escaped spaces)
    pdf_path = normalize_path(path_input)
    
    # Validate file
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return
    
    if not pdf_path.is_file():
        print(f"Error: Not a file: {pdf_path}")
        return
    
    if pdf_path.suffix.lower() != '.pdf':
        print(f"Warning: File extension is not .pdf: {pdf_path.suffix}")
        print("Continue anyway? [y/N]: ", end="")
        confirm = input().strip().lower()
        if confirm != 'y':
            print("Cancelled")
            return
    
    try:
        # Test file
        doc = test_read_pdf(str(pdf_path))
        
        print("\n" + "="*60)
        print("✅ TEST PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
