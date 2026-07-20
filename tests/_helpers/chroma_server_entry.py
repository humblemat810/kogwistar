"""Direct Python entry point for short-lived real Chroma test servers.

On Windows, the ``chroma.exe`` console-script launcher can exit after starting
its Python child.  A test harness that owns only the launcher then cannot
reliably terminate the actual server.  Invoking this module keeps the spawned
``Popen`` process attached to the server for its full lifetime.
"""

from __future__ import annotations

from chromadb.cli.cli import app


if __name__ == "__main__":
    app()
