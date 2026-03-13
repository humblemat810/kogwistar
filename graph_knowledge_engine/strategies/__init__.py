# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import (
    Protocol,
    runtime_checkable,
    List,
    Tuple,
    Any,
    Dict,
    Optional,
    Iterable,
)
from pydantic import BaseModel
from .types import EngineLike
from ..engine_core.models import (
    Node,
    Edge,
    AdjudicationVerdict,
    LLMMergeAdjudication,
    AdjudicationQuestionCode,
    Span,
    AdjudicationTarget,
)


from ..typing_interfaces import (
    NodeLike,
    EdgeLike,
)
from .proposer import CompositeProposer, VectorProposer
from .adjudicators import (
    LLMPairAdjudicatorImpl,
    LLMBatchAdjudicatorImpl,
    Adjudicator,
    IAdjudicator,
)
from .verifiers import DefaultVerifier, VerifierConfig
from .merge_policies import PreferExistingCanonical


__all__ = [
    "VectorProposer",
    "CompositeProposer",
    "LLMPairAdjudicatorImpl",
    "LLMBatchAdjudicatorImpl",
    "DefaultVerifier",
    "VerifierConfig",
    "PreferExistingCanonical",
    "EngineLike",
    "NodeLike",
    "Adjudicator",
    "IAdjudicator",
    "DefaultVerifierEdgeLike",
    "AdjudicationTarget",
]
