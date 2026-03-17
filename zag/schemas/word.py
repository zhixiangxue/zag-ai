"""
Word document schema (DOC / DOCX)
"""

from .document import PageableDocument


class Word(PageableDocument):
    """
    Microsoft Word document (DOC / DOCX).
    All paged-document logic lives in PageableDocument.
    """
