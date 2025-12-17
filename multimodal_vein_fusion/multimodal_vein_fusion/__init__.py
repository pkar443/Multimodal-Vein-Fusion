from __future__ import annotations

"""
Local (non-installed) import shim.

This repository uses a `src/` layout. When running from the project root without
installing, this package ensures `src/` is on `sys.path` and extends the package
search path so `python -m multimodal_vein_fusion.gui` works end-to-end.
"""

import sys
from pathlib import Path
from pkgutil import extend_path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir():
    _SRC_STR = str(_SRC)
    if _SRC_STR not in sys.path:
        sys.path.insert(0, _SRC_STR)

__path__ = extend_path(__path__, __name__)  # type: ignore[misc]

__all__ = ["__version__"]
__version__ = "0.1.0"

