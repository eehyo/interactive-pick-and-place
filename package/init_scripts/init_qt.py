"""
init_qt.py
"""
from __future__ import annotations

import sys
from typing import Optional

from package.qt.qt_widgets import * # type: ignore

def get_qt_app(argv: Optional[list[str]] = None,style: str = "Fusion"):
    """
    Get or create a QApplication instance (safe to call repeatedly).

    In Jupyter with `%gui qt`, you typically do not need to call app.exec_() manually.
    This function only ensures the app exists and optionally applies a style.
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        if argv is None:
            argv = sys.argv
        app = QApplication(argv)

    if style:
        try:
            app.setStyle(style)
        except Exception:
            pass

    return app

try:
    get_qt_app(style="Fusion")
except Exception:
    pass

print("[qt] package initialized.")
