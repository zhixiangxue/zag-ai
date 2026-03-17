"""
PDF document schema
"""

from .document import PageableDocument


class PDF(PageableDocument):
    """
    PDF document.  All paged-document logic lives in PageableDocument.
    This class exists as a concrete type so callers can do isinstance(doc, PDF)
    and so legacy code keeps working without changes.
    """
