"""
PlainText document schema (MD / TXT)
"""

from .document import Page, PageableDocument


class PlainText(PageableDocument):
    """
    Plain-text or Markdown document (MD / TXT).

    These formats have no native page structure.  The entire content is stored
    as a single synthetic Page(page_number=1) so the rest of the pipeline
    (page inference, locate_pages) can treat it uniformly.

    Factory method::

        doc = PlainText.from_text(
            content=text,
            doc_id=hash,
            metadata=metadata,
        )
    """

    @classmethod
    def from_text(
        cls,
        content: str,
        doc_id: str,
        metadata,
    ) -> "PlainText":
        """
        Build a PlainText document from raw text, wrapping it in a single synthetic page.
        """
        synthetic_page = Page(
            page_number=1,
            content=content,
            units=[],
            metadata={},
        )
        return cls(
            doc_id=doc_id,
            content=content,
            pages=[synthetic_page],
            metadata=metadata,
            units=[],
        )
