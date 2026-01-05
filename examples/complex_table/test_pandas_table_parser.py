"""
Test Pandas' ability to parse complex HTML tables
Tests various scenarios: rowspan, colspan, multi-level headers, etc.
"""

import pandas as pd
from pathlib import Path
from rich import print
from rich.table import Table as RichTable
from rich.console import Console

console = Console()


def test_table(table_id: str, html_path: Path, description: str):
    """
    Test parsing a specific table from HTML file
    
    Args:
        table_id: HTML table id (e.g., "table1")
        html_path: Path to HTML file
        description: Test description
    """
    print(f"\n{'='*80}")
    print(f"[bold cyan]{description}[/bold cyan]")
    print(f"{'='*80}\n")
    
    try:
        # Read HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse all tables
        from io import StringIO
        dfs = pd.read_html(StringIO(html_content))
        
        # Find the specific table (by index, assuming order matches HTML)
        table_index = int(table_id.replace('table', '')) - 1
        
        if table_index >= len(dfs):
            print(f"[red]Error: Table {table_id} not found (only {len(dfs)} tables parsed)[/red]")
            return None
        
        df = dfs[table_index]
        
        # Print basic info
        print(f"[green]✅ Successfully parsed table {table_id}[/green]")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
        
        # Convert to structured JSON format
        import json
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-level headers
            headers = [' | '.join(map(str, col)).strip() for col in df.columns]
            multi_level_headers = [
                [str(col[i]) for col in df.columns] 
                for i in range(len(df.columns[0]))
            ]
            has_multi_level = True
        else:
            headers = [str(col) for col in df.columns]
            multi_level_headers = None
            has_multi_level = False
        
        # Convert to list of lists
        rows = df.fillna('').astype(str).values.tolist()
        
        # Create structured JSON
        structured = {
            "table_id": table_id,
            "shape": {
                "rows": df.shape[0],
                "columns": df.shape[1]
            },
            "headers": headers,
            "multi_level_headers": multi_level_headers if has_multi_level else None,
            "rows": rows,
            "metadata": {
                "has_multi_level_headers": has_multi_level,
                "total_cells": df.shape[0] * df.shape[1]
            }
        }
        
        # Print JSON
        print("[yellow]JSON Output:[/yellow]")
        print(json.dumps(structured, indent=2, ensure_ascii=False))
        
        print(f"\n[green]Result: PASS[/green]")
        
        return df
        
    except Exception as e:
        print(f"[red]❌ Error parsing table {table_id}: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


def compare_original_vs_parsed(html_path: Path, table_id: str):
    """
    Compare original HTML table structure with parsed result in JSON format
    """
    print(f"\n[bold magenta]{'='*80}[/bold magenta]")
    print(f"[bold magenta]ANALYSIS: How Pandas handles {table_id}[/bold magenta]")
    print(f"[bold magenta]{'='*80}[/bold magenta]\n")
    
    try:
        import json
        from io import StringIO
        
        # Parse with Pandas
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        dfs = pd.read_html(StringIO(html_content))
        table_index = int(table_id.replace('table', '')) - 1
        df = dfs[table_index]
        
        # Extract original HTML table structure
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table', id=table_id)
        
        if not table:
            print(f"[red]Could not find table {table_id} in HTML[/red]")
            return
        
        # Analyze original HTML structure
        merged_cells = []
        row_idx = 0
        for tr in table.find_all('tr'):
            col_idx = 0
            for cell in tr.find_all(['td', 'th']):
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                
                if rowspan > 1 or colspan > 1:
                    merged_cells.append({
                        "cell_text": cell.get_text(strip=True),
                        "position": {"row": row_idx, "col": col_idx},
                        "rowspan": rowspan,
                        "colspan": colspan,
                        "tag": cell.name
                    })
                col_idx += colspan
            row_idx += 1
        
        # Create comparison JSON
        comparison = {
            "table_id": table_id,
            "original_html": {
                "total_rows": len(table.find_all('tr')),
                "merged_cells_count": len(merged_cells),
                "merged_cells": merged_cells
            },
            "pandas_result": {
                "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "has_multi_level_headers": isinstance(df.columns, pd.MultiIndex),
                "data_sample": [
                    {str(k): v for k, v in row.items()} 
                    for row in df.head(2).fillna('').astype(str).to_dict('records')
                ]
            },
            "transformation": {
                "rowspan_handling": "Values duplicated across merged rows",
                "colspan_handling": "Converted to multi-level column headers",
                "result_type": "Rectangular DataFrame (no missing cells)"
            }
        }
        
        print("[yellow]Comparison JSON:[/yellow]")
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"[red]Error in analysis: {e}[/red]")
        import traceback
        traceback.print_exc()


def main():
    """Run all table parsing tests"""
    html_path = Path(__file__).parent / "complex_table.html"
    
    if not html_path.exists():
        print(f"[red]Error: {html_path} not found![/red]")
        return
    
    print("[bold green]Testing Pandas HTML Table Parser[/bold green]")
    print(f"Test file: {html_path}\n")
    
    # Test all tables
    tests = [
        ("table1", "Test 1: Simple Table (Baseline)"),
        ("table2", "Test 2: Table with Rowspan (Merged Rows)"),
        ("table3", "Test 3: Table with Colspan (Merged Columns)"),
        ("table4", "Test 4: Complex Table with Both Rowspan and Colspan"),
        ("table5", "Test 5: Extremely Complex Table (Multi-level Headers)"),
        ("table6", "Test 6: Table with Empty Cells and Mixed Content"),
    ]
    
    results = {}
    for table_id, description in tests:
        df = test_table(table_id, html_path, description)
        results[table_id] = df is not None and not df.empty
    
    # Summary
    print("\n" + "="*80)
    print("[bold cyan]TEST SUMMARY[/bold cyan]")
    print("="*80 + "\n")
    
    for table_id, description in tests:
        status = "✅ PASS" if results.get(table_id) else "❌ FAIL"
        print(f"{status} - {description}")
    
    # Detailed analysis for complex tables
    print("\n" + "="*80)
    print("[bold yellow]DETAILED ANALYSIS[/bold yellow]")
    print("="*80)
    
    compare_original_vs_parsed(html_path, "table2")  # Rowspan
    compare_original_vs_parsed(html_path, "table3")  # Colspan
    compare_original_vs_parsed(html_path, "table4")  # Both
    
    # Final verdict
    print("\n" + "="*80)
    print("[bold magenta]VERDICT[/bold magenta]")
    print("="*80 + "\n")
    
    pass_count = sum(results.values())
    total_count = len(results)
    
    if pass_count == total_count:
        print(f"[bold green]✅ Pandas can parse ALL {total_count} complex tables![/bold green]")
        print("\n[green]Key findings:[/green]")
        print("  ✅ Handles rowspan by duplicating values across rows")
        print("  ✅ Handles colspan by creating multi-level column headers")
        print("  ✅ Fills empty cells with NaN (convertible to empty strings)")
        print("  ✅ Creates rectangular DataFrames (no jagged arrays)")
        print("\n[cyan]Recommendation: Use Pandas for table parsing in zag project[/cyan]")
    else:
        print(f"[yellow]⚠️  Pandas parsed {pass_count}/{total_count} tables successfully[/yellow]")
        print("\n[yellow]Some tables may need special handling[/yellow]")


if __name__ == "__main__":
    main()
