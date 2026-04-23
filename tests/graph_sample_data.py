from __future__ import annotations

from typing import Any

from kogwistar.engine_core.models import LLMGraphExtraction


def build_small_test_docs_nodes_edge_adjudcate() -> dict[str, Any]:
    """Return the legacy MCP adjudication sample bundle used by integration tests."""
    docs = {
        "DOC_A": (
            "In plant biology, many processes sustain life. "
            "Photosynthesis is a process used by plants to convert light energy into chemical energy. "
            "Chlorophyll is the molecule that absorbs sunlight. "
            "Within the leaves, specialized cells organize the chloroplasts. "
            "Plants perform photosynthesis in their leaves. "
            "Inside each chloroplast, stacks of thylakoid membranes called grana host the light-dependent reactions, "
            "where photons drive the formation of ATP and NADPH. "
            "These energy carriers then power the Calvin cycle in the stroma, fixing atmospheric CO2 into sugars using the enzyme Rubisco. "
            "Leaf anatomy supports this workflow: palisade mesophyll concentrates chloroplasts for maximal light capture, "
            "while spongy mesophyll and intercellular spaces facilitate gas diffusion. "
            "Stomata on the epidermis balance CO2 uptake with water conservation, opening and closing in response to light, humidity, and internal signals. "
            "Environmental factors-light intensity, temperature, CO2 concentration, and water availability-modulate overall photosynthetic rate. "
            "Different strategies evolved to mitigate photorespiration and arid stress: C3 plants fix carbon directly via Rubisco, "
            "C4 plants use a spatial separation with PEP carboxylase in bundle sheath cells, and CAM plants temporally separate uptake and fixation. "
            "Through these coordinated structures and pathways, photosynthesis underpins plant growth and, ultimately, most food webs."
        ),
        "DOC_B": (
            "Botanists distinguish several pigments in leaves. "
            "In most plants, chlorophyll a and chlorophyll b are present. "
            "The pigment chlorophyll gives leaves their green color. "
            "Both variants aid in harvesting light across different wavelengths. "
            "Chlorophyll a typically absorbs strongly in the blue-violet and red regions, while chlorophyll b extends coverage further into blue and a slightly different red band, "
            "broadening the effective spectrum plants can use. "
            "Accessory pigments such as carotenoids protect photosystems and capture light that chlorophylls miss, funneling energy into reaction centers by resonance transfer. "
            "The relative abundance of these pigments varies with species, developmental stage, and light environment-shade leaves often adjust pigment ratios to maximize efficiency. "
            "Seasonal changes alter pigment visibility: as chlorophyll degrades in autumn, carotenoids and, in some species, anthocyanins become more apparent, shifting leaf color. "
            "At the microscopic level, pigments are organized within protein complexes (photosystems I and II) embedded in thylakoid membranes, "
            "where the arrangement optimizes energy capture, minimizes photodamage, and supports electron transport."
        ),
        "DOC_C": (
            "Across the literature, the pigment chlorophyll is essential for photosynthesis, especially in terrestrial plants and algae. "
            "This role has been confirmed by numerous experiments. "
            "Classic action spectra align the rate of photosynthesis with wavelengths that chlorophyll absorbs, and historic demonstrations-such as Engelmann's-linked oxygen evolution to those bands. "
            "Mutational studies that reduce chlorophyll content depress photosynthetic performance, while stressors that damage chlorophyll or its binding proteins impair growth. "
            "Comparative work shows parallel solutions in other phototrophs: bacteriochlorophylls in anoxygenic bacteria illustrate how pigment chemistry adapts to distinct ecological niches. "
            "Modern techniques, including chlorophyll fluorescence measurements and satellite indices like NDVI, exploit chlorophyll's optical signatures to assess plant health and productivity at scales from leaves to landscapes. "
            "Biotechnological approaches aim to tune pigment composition and antenna size to minimize energy losses under high light, increase carbon gain, and enhance crop yields. "
            "Taken together, diverse lines of evidence establish chlorophyll as the core photochemical hub that initiates and regulates the flow of solar energy into biosynthetic pathways."
        ),
        "DOC_D": (
            "In human physiology, hemoglobin plays a central role. "
            "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues. "
            "Binding is mediated by iron within the heme groups. "
            "Structurally, adult hemoglobin (HbA) is a tetramer whose subunits exhibit cooperative binding, creating a sigmoidal oxygen dissociation curve that suits loading in the lungs and unloading in tissues. "
            "Physiological modulators-including pH and CO2 (the Bohr effect), temperature, and 2,3-bisphosphoglycerate-shift hemoglobin's affinity to match metabolic demand. "
            "Fetal hemoglobin (HbF) binds oxygen more tightly, facilitating transfer across the placenta, while variants such as sickle hemoglobin alter red-cell properties and clinical outcomes. "
            "Heme iron (Fe2+) reversibly binds O2 but can be blocked by carbon monoxide, which competes strongly at the same site; methemoglobin formation (Fe3+) reduces oxygen-carrying capacity. "
            "Beyond oxygen transport, hemoglobin contributes to CO2 carriage and acid-base buffering. "
            "When erythrocytes are recycled, heme is catabolized to bilirubin and iron is conserved-a tightly regulated process reflecting hemoglobin's systemic importance."
        ),
    }

    def _find_span(doc_id: str, excerpt: str) -> tuple[int, int]:
        text = docs[doc_id]
        idx = text.find(excerpt)
        if idx < 0:
            raise AssertionError(f"Exact excerpt not found in {doc_id}: {excerpt!r}")
        return idx, idx + len(excerpt)

    def _ref(doc_id: str, excerpt: str, method: str = "llm") -> dict[str, Any]:
        start_char, end_char = _find_span(doc_id, excerpt)
        return {
            "doc_id": doc_id,
            "collection_page_url": "N/A",
            "document_page_url": "N/A",
            "page_number": 1,
            "start_char": start_char,
            "end_char": end_char,
            "excerpt": docs[doc_id][start_char:end_char],
            "context_before": docs[doc_id][max(0, start_char - 40) : start_char],
            "context_after": docs[doc_id][
                end_char : min(len(docs[doc_id]), end_char + 40)
            ],
            "verification": {
                "method": method,
                "is_verified": True,
                "score": 1.0,
                "notes": "fixture",
            },
            "insertion_method": "llm_graph_extraction",
        }

    nodes = [
        {
            "id": "N_CHLORO",
            "label": "Chlorophyll",
            "type": "entity",
            "summary": "A green pigment found in plants.",
            "mentions": [
                {
                    "spans": [
                        _ref("DOC_A", "Chlorophyll is the molecule that absorbs sunlight."),
                        _ref("DOC_C", "chlorophyll is essential for photosynthesis"),
                    ]
                }
            ],
        },
        {
            "id": "N_CHLORO_ALIAS",
            "label": "Chlorophyll (pigment)",
            "type": "entity",
            "summary": "Alias name for the same pigment.",
            "mentions": [{"spans": [_ref("DOC_C", "the pigment chlorophyll is essential")]}],
        },
        {
            "id": "N_CHLORO_A",
            "label": "Chlorophyll a",
            "type": "entity",
            "summary": "One variant of chlorophyll.",
            "mentions": [{"spans": [_ref("DOC_B", "chlorophyll a and chlorophyll b are present.")]}],
        },
        {
            "id": "N_PHOTOSYN",
            "label": "Photosynthesis",
            "type": "entity",
            "summary": "Converts light energy to chemical energy.",
            "mentions": [
                {
                    "spans": [
                        _ref(
                            "DOC_A",
                            "Photosynthesis is a process used by plants to convert light energy",
                        ),
                        _ref(
                            "DOC_C",
                            "rate of photosynthesis with wavelengths that chlorophyll absorbs",
                        ),
                    ]
                }
            ],
        },
        {
            "id": "N_LEAVES",
            "label": "Leaves",
            "type": "entity",
            "summary": "Plant organs that host photosynthesis.",
            "mentions": [
                {
                    "spans": [
                        _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
                        _ref("DOC_B", "The pigment chlorophyll gives leaves their green color."),
                    ]
                }
            ],
        },
        {
            "id": "N_SUN",
            "label": "Sunlight",
            "type": "entity",
            "summary": "Incoming solar radiation.",
            "mentions": [{"spans": [_ref("DOC_A", "light energy into chemical energy")]}],
        },
        {
            "id": "N_HEMO",
            "label": "Hemoglobin",
            "type": "entity",
            "summary": "Oxygen transport protein in blood.",
            "mentions": [
                {
                    "spans": [
                        _ref(
                            "DOC_D",
                            "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues.",
                        )
                    ]
                }
            ],
        },
        {
            "id": "N_OXY",
            "label": "Oxygen",
            "type": "entity",
            "summary": "O2 molecule transported in blood.",
            "mentions": [
                {
                    "spans": [
                        _ref(
                            "DOC_D",
                            "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues.",
                        )
                    ]
                }
            ],
        },
        {
            "id": "N_PHOTO_REIFIED",
            "label": "Photosynthesis occurs in Leaves",
            "type": "entity",
            "summary": "Photosynthesis occurs in Leaves",
            "properties": {"signature_text": "occurs_in(Photosynthesis, Leaves)"},
            "mentions": [{"spans": [_ref("DOC_A", "Plants perform photosynthesis in their leaves.")]}],
        },
    ]

    edges = [
        {
            "id": "E_CHLORO_ABSORB",
            "label": "Chlorophyll absorbs sunlight",
            "type": "relationship",
            "summary": "Chlorophyll absorbs sunlight.",
            "relation": "absorbs",
            "source_ids": ["N_CHLORO"],
            "target_ids": ["N_SUN"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "mentions": [
                {
                    "spans": [
                        _ref("DOC_A", "Chlorophyll is the molecule that absorbs sunlight.")
                    ]
                }
            ],
            "properties": {"signature_text": "absorbs(Chlorophyll, Sunlight)"},
        },
        {
            "id": "E_PHOTO_LEAVES",
            "label": "Photosynthesis occurs in leaves",
            "type": "relationship",
            "summary": "Photosynthesis happens in leaves.",
            "relation": "occurs_in",
            "source_ids": ["N_PHOTOSYN"],
            "target_ids": ["N_LEAVES"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "mentions": [{"spans": [_ref("DOC_A", "Plants perform photosynthesis in their leaves.")]}],
            "properties": {"signature_text": "occurs_in(Photosynthesis, Leaves)"},
        },
        {
            "id": "E_PHOTO_LEAVES_DUP",
            "label": "Photosynthesis in leaves (duplicate)",
            "type": "relationship",
            "summary": "Duplicate of Photosynthesis->Leaves relation.",
            "relation": "occurs_in",
            "source_ids": ["N_PHOTOSYN"],
            "target_ids": ["N_LEAVES"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "mentions": [{"spans": [_ref("DOC_A", "Plants perform photosynthesis in their leaves.")]}],
            "properties": {"signature_text": "occurs_in(Photosynthesis, Leaves)"},
        },
        {
            "id": "E_HEMO_TRANSPORT",
            "label": "Hemoglobin transports oxygen",
            "type": "relationship",
            "summary": "Hemoglobin transports oxygen in blood.",
            "relation": "transports",
            "source_ids": ["N_HEMO"],
            "target_ids": ["N_OXY"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "mentions": [
                {
                    "spans": [
                        _ref(
                            "DOC_D",
                            "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues.",
                        )
                    ]
                }
            ],
            "properties": {"signature_text": "transports(Hemoglobin, Oxygen)"},
        },
    ]

    adjudication_pairs = {
        "positive": [
            ["N_CHLORO", "N_CHLORO_ALIAS"],
            ["N_PHOTO_REIFIED", "E_PHOTO_LEAVES"],
        ],
        "negative": [
            ["N_CHLORO", "N_HEMO"],
            ["N_CHLORO_A", "N_CHLORO_ALIAS"],
        ],
        "edge_positive": [["E_PHOTO_LEAVES", "E_PHOTO_LEAVES_DUP"]],
        "edge_negative": [["E_CHLORO_ABSORB", "E_HEMO_TRANSPORT"]],
        "cross_positive": [["N_PHOTO_REIFIED", "E_PHOTO_LEAVES"]],
        "cross_negative": [["N_HEMO", "E_CHLORO_ABSORB"]],
    }

    try:
        _ = LLMGraphExtraction.FromLLMSlice(
            {"nodes": nodes, "edges": edges}, insertion_method="fixture_sample"
        )
    except Exception as exc:
        raise AssertionError(
            f"Fixture small_test_docs_nodes_edge_adjudcate is not model-compatible: {exc}"
        ) from exc

    return {
        "docs": docs,
        "nodes": nodes,
        "edges": edges,
        "adjudication_pairs": adjudication_pairs,
    }
