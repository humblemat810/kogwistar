from __future__ import annotations

import uuid


import pytest

pytestmark = pytest.mark.ci_full

from kogwistar.engine_core.engine import (
    AliasBook,
    GraphKnowledgeEngine,
    base62_to_uuid,
    uuid_to_base62,
)
from kogwistar.engine_core.models import (
    AdjudicationVerdict,
    Document,
    Edge,
    Grounding,
    LLMGraphExtraction,
    MentionVerification,
    Node,
    Span,
)
