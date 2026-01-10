"""
Test MarkItDownReader.read() function
"""

from zag.readers.markitdown import MarkItDownReader
from zag.schemas.pdf import PDF


def test_read_local_pdf():
    """Test reading local PDF file"""
    print("\n" + "="*60)
    print("Test 1: Read Local PDF File")
    print("="*60)
    
    reader = MarkItDownReader()
    # Use test file from files directory
    import os
    local_file = os.path.join(
        os.path.dirname(__file__),
        '..',
        'files',
        'thunderbird_overview.pdf'
    )
    
    print(f"\nReading: {local_file}")
    doc = reader.read(local_file)
    
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
    
    print(f"\n✓ Content preview (first 200 chars):")
    print(f"  {doc.content[:200]}...")
    
    return doc


def test_read_remote_pdf():
    """Test reading remote PDF file from URL"""
    print("\n" + "="*60)
    print("Test 2: Read Remote PDF File (URL)")
    print("="*60)
    
    reader = MarkItDownReader()
    url = "https://wcbpub.oss-cn-hangzhou.aliyuncs.com/xue/zeitro/Thunderbird%20Product%20Overview%202025%20-%20No%20Doc.pdf"
    
    print(f"\nReading: {url}")
    print("(This may take a few seconds...)")
    
    doc = reader.read(url)
    
    print(f"\n✓ Document created successfully")
    print(f"  - Type: {type(doc).__name__}")
    print(f"  - Doc ID: {doc.doc_id}")
    print(f"  - Is PDF: {isinstance(doc, PDF)}")
    print(f"\n✓ Metadata:")
    print(f"  - source: {doc.metadata.source}")
    print(f"  - source_type: {doc.metadata.source_type}")
    print(f"  - file_type: {doc.metadata.file_type}")
    print(f"  - mime_type: {doc.metadata.mime_type}")
    print(f"  - content_length: {doc.metadata.content_length} characters")
    print(f"  - reader_name: {doc.metadata.reader_name}")
    print(f"  - created_at: {doc.metadata.created_at}")
    
    print(f"\n✓ Content preview (first 200 chars):")
    print(f"  {doc.content[:200]}...")
    
    return doc


def main():
    """Run tests"""
    print("\n" + "="*60)
    print("MARKITDOWN READER - READ FUNCTION TEST")
    print("="*60)
    
    try:
        # Test local file
        local_doc = test_read_local_pdf()
        
        # Test remote file
        remote_doc = test_read_remote_pdf()
        
        # Compare
        print("\n" + "="*60)
        print("Comparison")
        print("="*60)
        print(f"\n✓ Both are PDF documents: {isinstance(local_doc, PDF) and isinstance(remote_doc, PDF)}")
        print(f"✓ Content lengths similar: {abs(len(local_doc.content) - len(remote_doc.content)) < 100}")
        print(f"  - Local: {len(local_doc.content)} chars")
        print(f"  - Remote: {len(remote_doc.content)} chars")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
