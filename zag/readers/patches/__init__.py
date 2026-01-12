"""
Temporary patches for fixing reader output issues

🔧 This module contains temporary workarounds for known issues in upstream libraries.
   These patches should be removed once the issues are fixed upstream.
"""

from .simple_header_level_fixer import SimpleHeaderLevelFixer

__all__ = [
    "SimpleHeaderLevelFixer",
]
