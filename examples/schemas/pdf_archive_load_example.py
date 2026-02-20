"""
Example: Load PDF from archive format (interactive)

This example demonstrates how to load a previously dumped PDF archive.
Archives are created by PDF.dump() and contain:
- manifest.json: Metadata and stats
- content.md: Full document content
- metadata.json: Document metadata
- pages/: Per-page content files
- tables/: Table data (if available)

Use case: Share processed PDF results across teams/environments
"""
from pathlib import Path
from zag.schemas.pdf import PDF


def main():
    # Interactive input for archive path
    print("=" * 60)
    print("PDF Archive Loader")
    print("=" * 60)
    print("\nPlease enter the path to the PDF archive directory:")
    print("(Example: C:\\Users\\qu179\\.zag\\cache\\readers\\mineru\\01a1461e070f1308)")
    
    archive_path_str = input("\nArchive path: ").strip()
    
    if not archive_path_str:
        print("\n❌ Error: Archive path cannot be empty")
        return
    
    archive_path = Path(archive_path_str)
    
    if not archive_path.exists():
        print(f"\n❌ Error: Archive path does not exist: {archive_path}")
        return
    
    if not (archive_path / "manifest.json").exists():
        print(f"\n❌ Error: Invalid archive (manifest.json not found): {archive_path}")
        return
    
    # Load PDF from archive
    print(f"\nLoading PDF archive from: {archive_path}\n")
    
    try:
        pdf = PDF.load(archive_path)
    except Exception as e:
        print(f"\n❌ Error loading archive: {e}")
        return
    
    # Print basic info
    print("=" * 60)
    print("PDF Archive Info")
    print("=" * 60)
    print(f"Document ID: {pdf.doc_id}")
    print(f"Total pages: {len(pdf.pages)}")
    print(f"Content length: {len(pdf.content) if pdf.content else 0} characters")
    print(f"Units count: {len(pdf.units)}")
    
    # Print metadata if available
    if pdf.metadata:
        print(f"\nDocument Metadata:")
        print(f"  File name: {pdf.metadata.file_name}")
        print(f"  MD5: {pdf.metadata.md5}")
        print(f"  Content length: {pdf.metadata.content_length}")
        if pdf.metadata.custom:
            print(f"  Custom fields:")
            for key, value in pdf.metadata.custom.items():
                print(f"    - {key}: {value}")
    
    # Show page information
    print(f"\nPage Information:")
    print(f"  Total pages: {len(pdf.pages)}")
    if pdf.pages:
        print(f"  Page numbers: {[p.page_number for p in pdf.pages]}")
        
        # Show first page preview
        first_page = pdf.pages[0]
        print(f"\n  First page (#{first_page.page_number}):")
        content_preview = first_page.content[:300] if first_page.content else "No content"
        print(f"    Content preview: {content_preview}...")
        
        # Show last page preview
        if len(pdf.pages) > 1:
            last_page = pdf.pages[-1]
            print(f"\n  Last page (#{last_page.page_number}):")
            content_preview = last_page.content[:300] if last_page.content else "No content"
            print(f"    Content preview: {content_preview}...")
    
    # Show content preview
    if pdf.content:
        print(f"\nFull Content Preview (first 500 chars):")
        print("=" * 60)
        print(pdf.content[:500])
        print("=" * 60)
    
    # Example: Access specific page
    if len(pdf.pages) >= 3:
        page_3 = pdf.pages[2]  # 0-indexed
        print(f"\nAccessing Page 3:")
        print(f"  Page number: {page_3.page_number}")
        print(f"  Content length: {len(page_3.content) if page_3.content else 0}")
    
    # Example: Iterate through pages
    print(f"\nIterating through all pages:")
    for i, page in enumerate(pdf.pages[:5], 1):  # Show first 5 pages
        content_len = len(page.content) if page.content else 0
        print(f"  Page {page.page_number}: {content_len} characters")
    
    if len(pdf.pages) > 5:
        print(f"  ... and {len(pdf.pages) - 5} more pages")
    
    print(f"\n✓ Successfully loaded PDF archive!")


if __name__ == "__main__":
    main()
