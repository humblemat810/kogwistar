# -*- coding: utf-8 -*-
from __future__ import annotations

from collections.abc import Iterable
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from pydantic import BaseModel

from ..engine_core.models import (
    AdjudicationQuestionCode,
    AdjudicationTarget,
    AdjudicationVerdict,
    Edge,
    LLMMergeAdjudication,
    Node,
    Span,
)
from ..typing_interfaces import (
    EdgeLike,
    NodeLike,
)
from .adjudicators import (
    Adjudicator,
    IAdjudicator,
    LLMBatchAdjudicatorImpl,
    LLMPairAdjudicatorImpl,
)
from .merge_policies import PreferExistingCanonical
from .proposer import CompositeProposer, VectorProposer
from .types import EngineLike
from .verifiers import DefaultVerifier, VerifierConfig

__all__ = [
    "AdjudicationTarget",
    "Adjudicator",
    "CompositeProposer",
    "DefaultVerifier",
    "EngineLike",
    "IAdjudicator",
    "LLMBatchAdjudicatorImpl",
    "LLMPairAdjudicatorImpl",
    "NodeLike",
    "PreferExistingCanonical",
    "VectorProposer",
    "VerifierConfig",
]
