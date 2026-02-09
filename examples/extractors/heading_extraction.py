"""
Example: PDF Heading Extraction with Optional LLM Correction

Demonstrates how to extract headings from PDF using HeadingExtractor
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from zag.extractors.heading import HeadingExtractor

# Load environment variables
load_dotenv()


async def main():
    """Main execution"""
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip()
    
    # Clean up PowerShell drag-and-drop artifacts
    pdf_path = pdf_path.strip('"').strip("'")
    if pdf_path.startswith("& "):
        pdf_path = pdf_path[2:].strip().strip('"').strip("'")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Ask for LLM correction
    use_llm = input("Use LLM correction? (y/n, default: n): ").strip().lower() == 'y'
    
    print("\n" + "=" * 80)
    print(f"PDF: {pdf_path}")
    print(f"LLM Correction: {'Yes' if use_llm else 'No'}")
    print("=" * 80 + "\n")
    
    if use_llm:
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            return
        
        # Extract with LLM correction
        extractor = HeadingExtractor(
            llm_uri="openai/gpt-4o",
            api_key=api_key,
            llm_correction=True
        )
        
        print("Extracting headings with LLM correction...\n")
        headings = await extractor.aextract_from_pdf(pdf_path)
    else:
        # Extract without LLM
        extractor = HeadingExtractor()
        
        print("Extracting headings...\n")
        headings = extractor.extract_from_pdf(pdf_path)
    
    # Display results
    if not headings:
        print("No headings found.")
        return
    
    print(f"Found {len(headings)} headings\n")
    
    # Prepare output directory
    output_dir = Path("tmp")
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filename
    pdf_name = Path(pdf_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{pdf_name}_headings_{timestamp}.md"
    
    # Write to markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        for h in headings:
            # Write markdown heading
            md_heading = '#' * h['level'] + ' ' + h['text']
            f.write(md_heading + '\n\n')
    
    # Write hierarchical JSON tree
    json_file = output_dir / f"{pdf_name}_headings_tree_{timestamp}.json"
    tree = build_heading_tree(headings)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)
    
    print(f"Headings saved to: {output_file}")
    print(f"Hierarchical tree saved to: {json_file}")


def build_heading_tree(headings: list) -> list:
    """
    Build hierarchical tree structure from flat heading list
    
    Args:
        headings: Flat list of headings with 'level' and 'text' fields
        
    Returns:
        Nested tree structure
    """
    if not headings:
        return []
    
    root = []
    stack = [(0, root)]  # (level, children_list)
    
    for h in headings:
        level = h['level']
        node = {
            'text': h['text'],
            'level': level,
            'page': h['page'],
            'children': []
        }
        
        # Pop stack until we find the parent level
        while stack and stack[-1][0] >= level:
            stack.pop()
        
        # Add to parent's children
        if stack:
            stack[-1][1].append(node)
        else:
            root.append(node)
        
        # Push current node to stack
        stack.append((level, node['children']))
    
    return root


if __name__ == "__main__":
    asyncio.run(main())
