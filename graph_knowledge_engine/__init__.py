"""Legacy compatibility alias for old pickle/module paths.

Keep old `graph_knowledge_engine.*` imports alive by forwarding to `kogwistar.*`.
"""

from __future__ import annotations

from importlib import import_module
import sys

_KOGWISTAR = import_module("kogwistar")

# Re-export top-level package symbols where possible.
from kogwistar import *  # noqa: F401,F403

sys.modules.setdefault("graph_knowledge_engine", sys.modules[__name__])
sys.modules.setdefault("graph_knowledge_engine.kogwistar", _KOGWISTAR)

