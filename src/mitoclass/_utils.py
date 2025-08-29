from __future__ import annotations

import contextlib

from napari import Viewer


def status(message: str, msecs: int = 10000, viewer: Viewer | None = None):
    """
    Displays a message in the napari status bar (if a viewer is provided)
    and in the console.
    """
    if viewer is not None:
        with contextlib.suppress(Exception):
            viewer.window._qt_window.statusBar().showMessage(message, msecs)
    print(message)
