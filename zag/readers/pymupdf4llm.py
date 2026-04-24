"""
PyMuPDF4LLM reader — converts PDF to Markdown using pymupdf4llm.

High-accuracy local extraction (no LLM needed for basic use).
Returns PDF with real per-page structure (same as MinerUReader).

Optional table enhancement (enhance_tables=True):
    Complex tables (colspan/rowspan) are detected from the markdown output
    and re-extracted using Claude vision for accurate HTML representation.
    Simple tables remain as Markdown.

Dependencies:
    pymupdf4llm — pip install pymupdf4llm
    chak        — for Claude vision calls (only needed if enhance_tables=True)
"""

import asyncio
import base64
import concurrent.futures
import re
from pathlib import Path
from typing import Optional, Tuple, Union

from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseReader
from ..schemas import DocumentMetadata, Page
from ..schemas.pdf import PDF
from ..utils.hash import calculate_file_hash

console = Console()


# Claude model used for table vision extraction.
# Update to the latest/strongest model available on your account.
_CLAUDE_MODEL = "anthropic/claude-sonnet-4-5"

# Separator line pattern (|---|---|)
_SEP_RE = re.compile(r"^\|[-:| ]+\|$")
# Consecutive lines starting with |
_TABLE_BLOCK_RE = re.compile(r"(?:^\|[^\n]*\n?)+", re.MULTILINE)

# HTML table block regex — used only for extracting original strings (position-safe)
_HTML_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)


# Max concurrent Claude calls per document
_MAX_CONCURRENT = 5


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #

def _get_cache_dir(pdf_path: str) -> Path:
    """
    Return a content-addressed cache directory for this PDF.

    Path: {tmp}/zag_pdf_enhance_cache/{pdf_stem}_{md5}/
    Stored under the system temp directory so it works regardless of whether
    the PDF's parent directory is writable.
    """
    import tempfile
    pdf = Path(pdf_path)
    short_hash = calculate_file_hash(pdf)
    return Path(tempfile.gettempdir()) / "zag_pdf_enhance_cache" / f"{pdf.stem}_{short_hash}"


def _cache_page_path(cache_dir: Path, page_index: int) -> Path:
    """Return the cache file path for a single page (0-based index)."""
    return cache_dir / f"page_{page_index + 1:04d}.md"


# --------------------------------------------------------------------------- #

def extract_md_table_blocks(text: str) -> list[tuple[int, int, str]]:
    """
    Find all markdown table blocks in text and return their positions.

    Returns:
        List of (start_char, end_char, table_text) tuples, in document order.
    """
    return [(m.start(), m.end(), m.group()) for m in _TABLE_BLOCK_RE.finditer(text)]


def is_complex_table(table_text: str) -> bool:
    """
    Detect whether a markdown table likely contains true colspan or rowspan.

    Signals:
    - colspan: adjacent identical non-empty cells in any row
    - rowspan: at least 2 consecutive rows where the first cell is empty
              (single isolated empty-first-cell rows are layout artifacts, not merges)
    """
    lines = [
        line for line in table_text.strip().split("\n")
        if line.strip().startswith("|") and not _SEP_RE.match(line.strip())
    ]

    consecutive_empty_first = 0

    for line in lines:
        parts = line.split("|")
        cells = [c.strip() for c in parts[1:-1]]  # skip leading/trailing empty splits
        if not cells:
            continue

        # colspan: adjacent identical non-empty cells
        for i in range(len(cells) - 1):
            if cells[i] and cells[i] == cells[i + 1]:
                return True

        # rowspan: track consecutive rows with empty first cell
        if cells[0] == "":
            consecutive_empty_first += 1
            if consecutive_empty_first >= 2:
                return True
        else:
            consecutive_empty_first = 0

    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def _call_claude_page_async(
    pdf_path: str,
    page_index: int,
    reference_text: str,
    api_key: str,
) -> str:
    """
    Render a PDF page, send it to Claude together with the pymupdf4llm reference
    text, and get back the full page content as markdown with HTML tables.

    Args:
        pdf_path:       Path to the PDF file.
        page_index:     0-based page index.
        reference_text: pymupdf4llm's existing extraction for this page
                        (used so Claude can inherit correct heading structure).
        api_key:        Anthropic API key.

    Returns:
        Full page content as markdown string (headings in ##/###, tables as HTML).
    """
    import fitz
    import chak
    from chak import Image as ChakImage

    # Render the page as PNG
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    img_bytes = page.get_pixmap(dpi=200).tobytes("png")
    doc.close()

    b64 = base64.b64encode(img_bytes).decode()
    data_uri = f"data:image/png;base64,{b64}"

    prompt = (
        "Below is a preliminary text extraction of this PDF page for reference "
        "(the heading structure is likely correct):\n\n"
        f"{reference_text}\n\n"
        "---\n\n"
        "Now re-extract the full page from the image above. Rules:\n"
        "- Preserve the heading hierarchy from the reference (## and ### markers).\n"
        "- Extract ALL tables as HTML with correct colspan and rowspan for merged cells.\n"
        "- Tables may have multi-level headers (column headers spanning multiple rows) "
        "and cells that span multiple rows or columns — capture these precisely.\n"
        "- Table tags: <table>, <thead>, <tbody>, <tr>, <th>, <td> only. "
        "No styles, no class, no id attributes.\n"
        "- Keep all non-table text exactly as shown in the image.\n"
        "Return ONLY the page content, nothing else."
    )

    conv = chak.Conversation(_CLAUDE_MODEL, api_key=api_key)
    response = await conv.asend(
        prompt,
        attachments=[ChakImage(data_uri)],
        timeout=120,
    )
    return response.content.strip()


async def _process_page_async(
    pdf_path: str,
    page_index: int,
    reference_text: str,
    api_key: str,
    cache_dir: Path,
    semaphore: asyncio.Semaphore,
) -> str:
    """
    Process a single page: load from cache if available, otherwise call Claude.
    Saves result to cache after a successful Claude call.
    """
    cache_file = _cache_page_path(cache_dir, page_index)

    if cache_file.exists():
        console.print(f"  Page {page_index + 1}: loaded from cache")
        return cache_file.read_text(encoding="utf-8")

    async with semaphore:
        result = await _call_claude_page_async(pdf_path, page_index, reference_text, api_key)

    cache_file.write_text(result, encoding="utf-8")
    console.print(f"  Page {page_index + 1}: done ({len(result)} chars)")
    return result


async def _enhance_pages_async(
    pdf_path: str,
    pages_to_enhance: set[int],
    chunks: list,
    api_key: str,
    cache_dir: Path,
) -> dict[int, str]:
    """
    Run all page enhancements concurrently (up to _MAX_CONCURRENT at a time).

    Returns:
        Dict mapping 0-based page index → enhanced page text.
        Pages that fail are omitted (caller keeps original).
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    async def _one(i: int) -> tuple[int, str]:
        text = await _process_page_async(
            pdf_path, i, chunks[i]["text"], api_key, cache_dir, semaphore
        )
        return i, text

    results = await asyncio.gather(
        *[_one(i) for i in sorted(pages_to_enhance)],
        return_exceptions=True,
    )

    out: dict[int, str] = {}
    failures: list[Exception] = []
    for item in results:
        if isinstance(item, Exception):
            console.print(f"  A page failed: {item}")
            failures.append(item)
        else:
            page_idx, text = item
            out[page_idx] = text

    # If every page failed, it is almost certainly a network/API outage —
    # abort immediately so the caller can retry rather than silently indexing
    # unenhanced table content.
    if failures:
        raise RuntimeError(
            f"Claude enhancement failed for {len(failures)} page(s). "
            f"Last error: {failures[-1]}"
        )

    return out


def _run_enhance_concurrent(
    pdf_path: str,
    pages_to_enhance: set[int],
    chunks: list,
    api_key: str,
    cache_dir: Path,
) -> dict[int, str]:
    """Sync entry point for _enhance_pages_async."""
    coro = _enhance_pages_async(pdf_path, pages_to_enhance, chunks, api_key, cache_dir)

    # Detect whether we are already inside a running event loop.
    # If yes, we must run the coroutine in a separate thread to avoid deadlock.
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if in_loop:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=600)  # re-raises any exception from the coroutine
    else:
        return asyncio.run(coro)


def call_claude_page(
    pdf_path: str,
    page_index: int,
    reference_text: str,
    api_key: str,
) -> str:
    """
    Sync wrapper around _call_claude_page_async.

    Works whether or not we're already inside a running event loop.
    """
    coro = _call_claude_page_async(pdf_path, page_index, reference_text, api_key)
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=150)
    except RuntimeError:
        return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Markdown → HTML table conversion
# --------------------------------------------------------------------------- #

def _md_row_cells(row: str, tag: str) -> str:
    """Convert a markdown table row string to HTML cell elements."""
    parts = row.split("|")
    cells = [c.strip() for c in parts[1:-1]]
    return "".join(f"<{tag}>{c}</{tag}>" for c in cells)


def _md_table_to_html(table_text: str) -> str:
    """Convert a single markdown table block to a plain HTML table."""
    lines = [l for l in table_text.strip().split("\n") if l.strip().startswith("|")]
    if not lines:
        return table_text

    sep_idx = next(
        (i for i, l in enumerate(lines) if _SEP_RE.match(l.strip())), None
    )

    if sep_idx is None:
        rows = "\n".join(f"<tr>{_md_row_cells(l, 'td')}</tr>" for l in lines)
        return f"<table>\n<tbody>\n{rows}\n</tbody>\n</table>"

    head_rows = "\n".join(
        f"<tr>{_md_row_cells(l, 'th')}</tr>" for l in lines[:sep_idx]
    )
    body_rows = "\n".join(
        f"<tr>{_md_row_cells(l, 'td')}</tr>" for l in lines[sep_idx + 1:]
    )
    return (
        f"<table>\n<thead>\n{head_rows}\n</thead>\n"
        f"<tbody>\n{body_rows}\n</tbody>\n</table>"
    )


def _convert_md_tables_to_html(text: str) -> str:
    """Replace all markdown table blocks in text with HTML tables."""
    return _TABLE_BLOCK_RE.sub(lambda m: _md_table_to_html(m.group()), text)


# --------------------------------------------------------------------------- #
# Cross-page table merging
# --------------------------------------------------------------------------- #

def _normalize_html(text: str) -> str:
    """Normalize HTML through bs4 so string matching is consistent."""
    from bs4 import BeautifulSoup
    return str(BeautifulSoup(text, "html.parser"))


def _find_html_tables(text: str) -> list[str]:
    """
    Return HTML strings of all TOP-LEVEL <table> blocks.
    bs4 handles nested tables correctly — inner tables are NOT returned
    as separate items, only their outermost ancestor is.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    return [
        str(t)
        for t in soup.find_all("table")
        if t.find_parent("table") is None
    ]


def _get_table_last_rows(html: str, n: int = 3) -> str:
    """Return last n <tr> rows from an HTML table (for LLM context)."""
    from bs4 import BeautifulSoup
    rows = BeautifulSoup(html, "html.parser").find_all("tr")
    return "\n".join(str(r) for r in rows[-n:]) if rows else html[:500]


def _get_table_first_rows(html: str, n: int = 3) -> str:
    """Return first n <tr> rows from an HTML table (for LLM context)."""
    from bs4 import BeautifulSoup
    rows = BeautifulSoup(html, "html.parser").find_all("tr")
    return "\n".join(str(r) for r in rows[:n]) if rows else html[:500]


def _append_table_rows(table_a: str, table_b: str) -> str:
    """
    Append body rows from table_b into table_a.
    Drops table_b's <thead> when it is identical to table_a's (repeated page header).
    """
    from bs4 import BeautifulSoup
    soup_a = BeautifulSoup(table_a, "html.parser")
    soup_b = BeautifulSoup(table_b, "html.parser")
    ta = soup_a.find("table")
    tb = soup_b.find("table")
    if not ta or not tb:
        return table_a
    thead_a = ta.find("thead", recursive=False)
    thead_b = tb.find("thead", recursive=False)
    tbody_a = ta.find("tbody", recursive=False) or ta
    tbody_b = tb.find("tbody", recursive=False)
    # Three cases:
    # 1. thead_b == thead_a: repeated page header — skip thead_b, keep only tbody_b rows
    # 2. thead_b exists but differs from thead_a: it is a continuation row (not a real
    #    header), treat its <tr> rows as body rows so they are NOT dropped
    # 3. No thead_b: take all rows from tbody_b (or whole table)
    repeated_header = (
        thead_b is not None
        and thead_a is not None
        and thead_a.get_text(strip=True) == thead_b.get_text(strip=True)
    )
    if repeated_header:
        rows = tbody_b.find_all("tr", recursive=False) if tbody_b else []
    elif thead_b is not None:
        # thead_b rows are continuation content — include them before tbody rows
        rows = thead_b.find_all("tr", recursive=False)
        rows += tbody_b.find_all("tr", recursive=False) if tbody_b else []
    else:
        rows = (
            tbody_b.find_all("tr", recursive=False)
            if tbody_b
            else tb.find_all("tr", recursive=False)
        )
    for row in rows:
        tbody_a.append(row)
    return str(soup_a)


def _extract_leading_fragment(content: str) -> tuple[str, str] | None:
    """
    Detect a dangling table fragment at the start of a page.

    When Claude processes a page that begins mid-table (no opening <table> tag),
    the page starts with tags like </td>, </tr>, <tr>, etc.
    This function extracts that fragment so it can be merged into the previous page.

    Returns (fragment_html, remaining_content) or None.
    """
    from bs4 import BeautifulSoup
    stripped = content.lstrip()
    if not re.match(r"</?(?:tr|td|tbody|thead|th)\b", stripped, re.IGNORECASE):
        return None
    # Partial HTML: bs4 can't find the boundary directly since the opening
    # <table> tag is on a previous page.  Wrap in a dummy outer table so bs4
    # can parse it, then measure how many characters that reconstructed table
    # spans in the original stripped text by counting depth.
    #
    # We still use a simple tag counter here — but purely for POSITION finding,
    # not for structural parsing (which bs4 handles everywhere else).
    depth = 1  # we're already inside one outer table
    tag_re = re.compile(r"<(/?)table\b", re.IGNORECASE)
    for m in tag_re.finditer(stripped):
        if not m.group(1):   # nested <table> opens
            depth += 1
        else:                # </table> closes
            depth -= 1
            if depth == 0:
                end = stripped.find(">", m.end()) + 1
                fragment = stripped[:end]
                remaining = stripped[end:]
                # Use bs4 to extract only the <tr> rows from the fragment
                # (validate it's parseable before returning)
                rows = BeautifulSoup(fragment, "html.parser").find_all("tr")
                if not rows:
                    return None
                return fragment, remaining
    return None


def _merge_dangling_fragment(table_html: str, fragment: str) -> str:
    """Extract <tr> rows from a dangling fragment and append them into table_html."""
    from bs4 import BeautifulSoup
    rows = BeautifulSoup(fragment, "html.parser").find_all("tr")
    if not rows:
        return table_html
    soup = BeautifulSoup(table_html, "html.parser")
    tbody = soup.find("tbody") or soup.find("table")
    if not tbody:
        return table_html
    for row in rows:
        tbody.append(row)
    return str(soup)


def _continue_last_cell(table_a: str, table_b: str) -> str:
    """
    Merge strategy for CONTINUE_CELL: the first block of table_b is the
    continuation of the last cell in table_a's last row.

    Steps:
    1. Extract the text content from table_b's <thead> (the continuation text).
    2. Append that text to the last <td> of table_a's last row.
    3. Append table_b's <tbody> rows as new rows to table_a.
    """
    from bs4 import BeautifulSoup
    soup_a = BeautifulSoup(table_a, "html.parser")
    soup_b = BeautifulSoup(table_b, "html.parser")
    ta = soup_a.find("table")
    tb = soup_b.find("table")
    if not ta or not tb:
        return table_a

    tbody_a = ta.find("tbody", recursive=False) or ta
    thead_b = tb.find("thead", recursive=False)
    tbody_b = tb.find("tbody", recursive=False)

    # Find the last row of table_a
    last_row = None
    for tr in tbody_a.find_all("tr", recursive=False):
        last_row = tr

    # Find the first (continuation) row of table_b — prefer thead, fall back to tbody
    first_row_b = None
    if thead_b:
        rows = thead_b.find_all("tr", recursive=False)
        if rows:
            first_row_b = rows[0]

    # Merge cell-by-cell: each column of first_row_b is appended to the
    # corresponding column of last_row (same position index)
    if last_row is not None and first_row_b is not None:
        cells_a = last_row.find_all(["td", "th"], recursive=False)
        cells_b = first_row_b.find_all(["td", "th"], recursive=False)
        for idx, cell_b in enumerate(cells_b):
            if cell_b.get_text(strip=True) and idx < len(cells_a):
                fragment = BeautifulSoup(cell_b.decode_contents(), "html.parser")
                for child in list(fragment.children):
                    cells_a[idx].append(child)

    # Append remaining tbody_b rows as new rows
    if tbody_b:
        for row in tbody_b.find_all("tr", recursive=False):
            tbody_a.append(row)

    return str(soup_a)


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    reraise=True,
)
async def _ask_should_merge_async(tail_a: str, head_b: str, api_key: str) -> str:
    """
    Ask Claude how to merge two HTML table fragments from consecutive PDF pages.

    Returns one of:
    - "NO"           : different tables, do not merge
    - "NEW_ROW"      : page N+1 starts a new row of the same table
    - "CONTINUE_CELL": page N+1 continues the last cell of page N's last row
    """
    import chak
    from pydantic import BaseModel as _BM, Field as _F
    from typing import Literal as _L
    
    class _MergeDecision(_BM):
        """Merge decision for two consecutive PDF table fragments."""
        merge: _L["NO", "NEW_ROW", "CONTINUE_CELL"] = _F(
            description=(
                "NO: different tables. "
                "NEW_ROW: page N+1 starts a new row of the same table. "
                "CONTINUE_CELL: page N+1 continues the last cell of page N's last row "
                "(the cell content was split across the page break)."
            )
        )
    
    prompt = (
        "Two HTML table fragments from consecutive PDF pages:\n\n"
        "--- END OF PAGE N ---\n"
        f"{tail_a}\n\n"
        "--- START OF PAGE N+1 ---\n"
        f"{head_b}\n\n"
        "Determine the relationship between these two fragments."
    )
    conv = chak.Conversation("anthropic/claude-haiku-4-5", api_key=api_key)
    result = await conv.asend(prompt, returns=_MergeDecision, timeout=30)
    return result.merge if result else "NO"


def _run_should_merge(tail_a: str, head_b: str, api_key: str) -> str:
    """Sync wrapper around _ask_should_merge_async. Returns 'NO', 'NEW_ROW', or 'CONTINUE_CELL'."""
    coro = _ask_should_merge_async(tail_a, head_b, api_key)
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=300)
    except RuntimeError:
        return asyncio.run(coro)


def _has_heading_before_table(content: str, table_html: str) -> bool:
    """Return True if a markdown heading (##, ###, etc.) appears before the table."""
    pos = content.find(table_html)
    if pos == -1:
        return False
    return bool(re.search(r"^#{1,6}\s", content[:pos], re.MULTILINE))


def _post_process_pages(
    pages: list[Page], api_key: str
) -> tuple[list[Page], list[str]]:
    """
    Post-process all pages after extraction and Claude table enhancement:

    1. Convert any remaining markdown tables to HTML (applied to every page).
    2. Linear stateful scan: detect and merge tables that span page boundaries.

    Returns:
    - normalized_pages: one Page per original PDF page, content = MD→HTML
      normalized but NOT modified by cross-page merges.  Used for per-page
      dump so each file corresponds 1-to-1 with the original PDF page.
    - merge_contents: per-page strings after merging (some pages become empty
      when their table was absorbed into a previous page).  Used to build
      full_content and span offsets in content.md.
    """
    # Step 1: normalize each page's content independently (MD→HTML + bs4).
    # This is the canonical per-page view — never modified by merge logic.
    normalized = [
        _normalize_html(_convert_md_tables_to_html(p.content)) for p in pages
    ]

    # Step 2: working copy for merge — this IS modified.
    contents = list(normalized)

    # Step 2: linear scan with "open reference" state
    # open_ref_idx: page index holding the accumulated (possibly multi-page) table
    open_ref_idx: Optional[int] = None
    n_merged = 0

    i = 0
    while i < len(contents):
        # --- Dangling fragment check (no <table> needed) ---
        if open_ref_idx is not None:
            dangling = _extract_leading_fragment(contents[i])
            if dangling:
                fragment, remaining = dangling
                ref_tables = _find_html_tables(contents[open_ref_idx])
                if ref_tables:
                    console.print(
                        f"  Dangling fragment: page {open_ref_idx + 1} <- page {i + 1} (auto-merge)"
                    )
                    merged = _merge_dangling_fragment(ref_tables[-1], fragment)
                    contents[open_ref_idx] = contents[open_ref_idx].replace(
                        ref_tables[-1], merged, 1
                    )
                    contents[i] = remaining
                    n_merged += 1
                    # Re-evaluate page i with the fragment removed
                    continue  # don't advance i yet; re-check for more tables

        tables = _find_html_tables(contents[i])

        if not tables:
            i += 1
            continue

        first_table_html = tables[0]

        if open_ref_idx is not None:
            # A heading before the first table means a new section — close chain
            if _has_heading_before_table(contents[i], first_table_html):
                console.print(
                    f"  Page {i + 1}: heading before table → new section, close chain"
                )
                open_ref_idx = None
                # Fall through: treat this page as a new open_ref candidate
            else:
                ref_tables = _find_html_tables(contents[open_ref_idx])
                if ref_tables:
                    tail_a = _get_table_last_rows(ref_tables[-1], n=3)
                    head_b = _get_table_first_rows(first_table_html, n=3)

                    console.print(
                        f"  Merge check: page {open_ref_idx + 1} <- page {i + 1} ... ",
                        end="",
                    )
                    decision = _run_should_merge(tail_a, head_b, api_key)
                    console.print(decision)

                    if decision in ("NEW_ROW", "CONTINUE_CELL"):
                        if decision == "CONTINUE_CELL":
                            merged = _continue_last_cell(ref_tables[-1], first_table_html)
                        else:
                            merged = _append_table_rows(ref_tables[-1], first_table_html)
                        contents[open_ref_idx] = contents[open_ref_idx].replace(
                            ref_tables[-1], merged, 1
                        )
                        contents[i] = contents[i].replace(first_table_html, "", 1)
                        n_merged += 1

                        # If page i still has tables after removal, it becomes new ref
                        if _find_html_tables(contents[i]):
                            open_ref_idx = i

                        i += 1
                        continue
                    else:
                        open_ref_idx = None
                        # Fall through: treat this page as new open_ref candidate
                else:
                    open_ref_idx = None

        # This page has tables and no active open_ref — start a new merge chain
        open_ref_idx = i
        i += 1

    if n_merged > 0:
        console.print(f"  Cross-page table merge: {n_merged} page boundaries merged")
    else:
        console.print("  Cross-page table merge: no cross-page tables detected")

    normalized_pages = [
        Page(page_number=p.page_number, content=normalized[i], units=p.units)
        for i, p in enumerate(pages)
    ]
    return normalized_pages, contents


# --------------------------------------------------------------------------- #
# Reader class
# --------------------------------------------------------------------------- #

class PyMuPDF4LLMReader(BaseReader):
    """
    Reader using pymupdf4llm for high-accuracy PDF → Markdown conversion.

    Returns PDF with real per-page structure (same output shape as MinerUReader).

    When enhance_tables=True:
        1. Extract full page content via pymupdf4llm (fast, no LLM)
        2. Detect complex markdown tables on each page
        3. Re-extract each complex table via Claude vision → proper HTML
        4. Replace the markdown table block with the HTML output

    Usage::

        # Basic (pure local, no LLM)
        reader = PyMuPDF4LLMReader()
        doc = reader.read("path/to/file.pdf")

        # With table enhancement
        reader = PyMuPDF4LLMReader(
            enhance_tables=True,
            anthropic_api_key="sk-ant-...",
        )
        doc = reader.read("path/to/file.pdf")
    """

    def __init__(
        self,
        enhance_tables: bool = False,
        anthropic_api_key: Optional[str] = None,
    ):
        """
        Args:
            enhance_tables:    Detect complex tables and re-extract via Claude vision.
            anthropic_api_key: Required when enhance_tables=True.
        """
        self.enhance_tables = enhance_tables
        self.anthropic_api_key = anthropic_api_key

        if enhance_tables and not anthropic_api_key:
            raise ValueError("anthropic_api_key is required when enhance_tables=True")

    def read(self, source: Union[str, Path], page_range: Optional[Tuple[int, int]] = None) -> PDF:
        """
        Read a PDF file and return its content as a structured PDF document.

        Args:
            source: Local file path to a PDF file.
            page_range: Optional page range as (start_page, end_page).
                        Page numbers are 1-based and inclusive.
                        Example: (1, 50) reads pages 1 through 50.
                        None means read all pages (default).
                        Range exceeding actual pages is auto-adjusted.

        Returns:
            PDF document with real per-page structure.

        Raises:
            ImportError: If pymupdf4llm is not installed.
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a PDF or page_range is invalid.
        """
        try:
            import pymupdf4llm
        except ImportError:
            raise ImportError("pymupdf4llm is required. Run: pip install pymupdf4llm")

        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"PyMuPDF4LLMReader only supports .pdf, got: {file_path.suffix}"
            )

        # Resolve page_range to a 0-based pages list for pymupdf4llm
        pages_arg: Optional[list[int]] = None
        if page_range is not None:
            start, end = page_range
            if start < 1 or end < start:
                raise ValueError(
                    f"Invalid page_range {page_range}: must satisfy 1 <= start <= end"
                )
            import fitz
            total = fitz.open(str(file_path)).page_count
            # Auto-adjust end to actual page count (no error, same as MinerU)
            end = min(end, total)
            pages_arg = list(range(start - 1, end))  # 0-based

        # page_chunks=True → list of {"metadata": {"page": N, ...}, "text": "..."}
        chunks = pymupdf4llm.to_markdown(
            str(file_path),
            page_chunks=True,
            **(dict(pages=pages_arg) if pages_arg is not None else {}),
        )

        if self.enhance_tables:
            chunks = self._enhance_all_pages(str(file_path), chunks)

        # Build raw Page objects (no span offsets yet — content may change after merge)
        raw_pages: list[Page] = []
        for i, chunk in enumerate(chunks):
            page_num = chunk["metadata"].get("page", i + 1)
            raw_pages.append(Page(
                page_number=page_num,
                content=chunk["text"],
                units=[],
            ))

        # Post-process: MD tables → HTML and cross-page table merging
        # normalized_pages: per-page content (unchanged by merges) → for page dump
        # merge_contents:   per-page content after merges → for content.md / spans
        if self.enhance_tables:
            normalized_pages, merge_contents = _post_process_pages(
                raw_pages, self.anthropic_api_key
            )
        else:
            normalized_pages = raw_pages
            merge_contents = [p.content for p in raw_pages]

        # Build final Page objects.
        # content = original per-page content (for dump).
        # span   = character offset into the merged full_content.
        pages: list[Page] = []
        all_parts: list[str] = []
        for page, merged_content in zip(normalized_pages, merge_contents):
            page_start = len("\n\n".join(all_parts)) if all_parts else 0
            if merged_content.strip():
                all_parts.append(merged_content)
            page_end = len("\n\n".join(all_parts))
            pages.append(Page(
                page_number=page.page_number,
                content=page.content,
                units=[],
                metadata={"span": (page_start, page_end)},
            ))

        full_content = "\n\n".join(all_parts)
        file_hash = calculate_file_hash(file_path)

        metadata = DocumentMetadata(
            source=str(file_path),
            source_type="local",
            file_type="pdf",
            file_name=file_path.name,
            file_size=file_path.stat().st_size,
            file_extension=".pdf",
            md5=file_hash,
            content_length=len(full_content),
            mime_type="application/pdf",
            reader_name="PyMuPDF4LLMReader",
        )

        return PDF(
            doc_id=metadata.md5,
            content=full_content,
            metadata=metadata,
            pages=pages,
        )

    # ----------------------------------------------------------------------- #
    # Table enhancement
    # ----------------------------------------------------------------------- #

    def _enhance_all_pages(self, pdf_path: str, chunks: list) -> list:
        """
        Enhance complex-table pages via Claude vision.

        - Pre-scans all pages to find targets before any API call.
        - Runs Claude calls concurrently (up to _MAX_CONCURRENT at a time).
        - Caches each page result as a .md file; resumes automatically on retry.
        """
        import fitz

        # Pre-scan: collect pages that need Claude
        pages_to_enhance: set[int] = set()
        for i, chunk in enumerate(chunks):
            md_tables = extract_md_table_blocks(chunk["text"])
            if any(is_complex_table(t) for _, _, t in md_tables):
                pages_to_enhance.add(i)

        if not pages_to_enhance:
            console.print("  No complex tables detected, skipping enhancement.")
            return chunks

        sorted_pages = sorted(p + 1 for p in pages_to_enhance)  # 1-based for display
        console.print(f"  Pages to send to Claude: {sorted_pages}")

        # Prepare cache directory
        cache_dir = _get_cache_dir(pdf_path)
        cache_dir.mkdir(parents=True, exist_ok=True)

        page_count = fitz.open(pdf_path).page_count
        valid_pages = {i for i in pages_to_enhance if i < page_count}

        # Run all target pages concurrently (with per-page caching)
        results = _run_enhance_concurrent(
            pdf_path, valid_pages, chunks, self.anthropic_api_key, cache_dir
        )

        # Build enhanced chunk list
        enhanced: list = []
        for i, chunk in enumerate(chunks):
            if i in results:
                new_chunk = dict(chunk)
                new_chunk["text"] = results[i]
                enhanced.append(new_chunk)
            else:
                if i in valid_pages:
                    console.print(f"  Page {i + 1}: failed, keeping original")
                enhanced.append(chunk)

        n_replaced = len(results)
        console.print(
            f"  Enhancement complete: {n_replaced}/{len(valid_pages)} pages replaced"
        )
        return enhanced
