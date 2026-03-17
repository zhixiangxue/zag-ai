#!/usr/bin/env python3
"""
Test Free Composition - Verify Order-Independent Splitter Composition

Tests that splitters can be composed in any order and still work correctly.
This verifies the unified interface design where each splitter can process
both Documents and list[BaseUnit].
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zag.splitters import MarkdownHeaderSplitter, TextSplitter, RecursiveMergingSplitter
from zag.schemas.plain import PlainText
from zag.schemas import DocumentMetadata
from zag.utils.hash import calculate_string_hash


def create_test_doc():
    """Create a test document with varied section sizes"""
    content = """# Section 1
Short content here.

## Subsection 1.1
A bit more content in this subsection with enough text to make it interesting.

# Section 2
This is a much longer section that will definitely exceed our token limits. """ + ("More text. " * 200) + """

## Subsection 2.1
Another subsection with moderate length content.

# Section 3
Short ending section.
"""
    
    metadata = DocumentMetadata(
        source="test",
        source_type="inline",
        file_type="markdown",
        content_length=len(content)
    )
    
    return PlainText.from_text(
        content=content,
        doc_id=calculate_string_hash(content),
        metadata=metadata,
    )


def analyze_result(name: str, units):
    """Analyze and print results"""
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    tokens = [len(tokenizer.encode(u.content)) for u in units]
    
    print(f"\n{name}:")
    print(f"  Units: {len(units)}")
    print(f"  Token range: {min(tokens)}-{max(tokens)}")
    print(f"  Average: {sum(tokens) // len(tokens)}")
    
    return {
        'count': len(units),
        'min': min(tokens),
        'max': max(tokens),
        'avg': sum(tokens) // len(tokens)
    }


def main():
    print("="*70)
    print("🧪 Free Composition Test - Order Independence")
    print("="*70)
    
    doc = create_test_doc()
    print(f"\n📄 Test document created ({len(doc.content)} chars)")
    
    # Test different orderings
    test_cases = [
        ("Order 1: Header → Text → Merge", 
         MarkdownHeaderSplitter() | TextSplitter(1200) | RecursiveMergingSplitter(800)),
        
        ("Order 2: Header → Merge → Text",
         MarkdownHeaderSplitter() | RecursiveMergingSplitter(800) | TextSplitter(1200)),
        
        ("Order 3: Text → Header → Merge",
         TextSplitter(1200) | MarkdownHeaderSplitter() | RecursiveMergingSplitter(800)),
        
        ("Order 4: Text → Merge → Header",
         TextSplitter(1200) | RecursiveMergingSplitter(800) | MarkdownHeaderSplitter()),
        
        ("Order 5: Merge → Header → Text",
         RecursiveMergingSplitter(800) | MarkdownHeaderSplitter() | TextSplitter(1200)),
        
        ("Order 6: Merge → Text → Header",
         RecursiveMergingSplitter(800) | TextSplitter(1200) | MarkdownHeaderSplitter()),
    ]
    
    results = {}
    
    print("\n" + "="*70)
    print("Testing All Possible Orderings...")
    print("="*70)
    
    for name, pipeline in test_cases:
        try:
            units = doc.split(pipeline)
            stats = analyze_result(name, units)
            results[name] = {'success': True, 'stats': stats}
        except Exception as e:
            print(f"\n{name}:")
            print(f"  ❌ FAILED: {str(e)}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "="*70)
    print("📊 Summary")
    print("="*70)
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    
    print(f"\nSuccess Rate: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✅ ALL ORDERINGS WORK! Free composition verified!")
        
        # Show statistics comparison
        print("\n" + "="*70)
        print("Statistics Comparison")
        print("="*70)
        print(f"{'Order':<40} {'Units':<8} {'Min':<8} {'Max':<8} {'Avg':<8}")
        print("-"*70)
        
        for name, result in results.items():
            if result['success']:
                stats = result['stats']
                short_name = name.split(':')[0]
                print(f"{short_name:<40} {stats['count']:<8} {stats['min']:<8} {stats['max']:<8} {stats['avg']:<8}")
        
        print("\n💡 Key Insight:")
        print("   Different orderings produce different results, but all work correctly!")
        print("   Developers can choose the order that best fits their use case.")
    else:
        print("\n❌ SOME ORDERINGS FAILED:")
        for name, result in results.items():
            if not result['success']:
                print(f"  - {name}: {result['error']}")
    
    # Practical recommendations
    print("\n" + "="*70)
    print("🎯 Practical Recommendations")
    print("="*70)
    print("""
For typical RAG use cases, recommended orderings:

1. Header → Text → Merge (Standard)
   - Best for: Documents with clear structure
   - Flow: Structure → Size control → Optimization
   
2. Text → Merge → Header (Size-first)
   - Best for: Unstructured documents
   - Flow: Size control → Optimization → Structure extraction
   
3. Merge → Text → Header (Optimization-first)
   - Best for: Pre-split units
   - Flow: Optimization → Size control → Structure refinement
""")


if __name__ == "__main__":
    try:
        import tiktoken
    except ImportError:
        print("❌ tiktoken is required")
        print("Run: pip install tiktoken")
        sys.exit(1)
    
    main()
