"""Lightweight graph types shared across public and engine-core modules."""

from typing import Literal

# Keep structural graph tokens outside engine_core to avoid loading its package
# facade when lightweight protocol modules need this alias.
GraphEntityKind = Literal["node", "edge"]
