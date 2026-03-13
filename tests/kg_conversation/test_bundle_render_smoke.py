from pathlib import Path


def test_bundle_contains_d3_rendering_logic(tmp_path, seeded_kg_and_conversation):
    """
    Smoke test only.

    Purpose:
    - Ensure the real d3.html template is included in the bundle.
    - Do NOT execute JavaScript.
    - Do NOT validate visual correctness.
    """
    kg_engine, conv_engine, *_ = seeded_kg_and_conversation
    from graph_knowledge_engine.utils.kge_debug_dump import dump_paired_bundles

    template_html = Path("graph_knowledge_engine/templates/d3.html").read_text(
        encoding="utf-8"
    )
    out_dir = tmp_path / "bundle"

    dump_paired_bundles(
        kg_engine=kg_engine,
        conversation_engine=conv_engine,
        template_html=template_html,
        out_dir=out_dir,
    )

    html = (out_dir / "conversation.bundle.html").read_text(encoding="utf-8")
    import os

    os.startfile(str(out_dir))

    # Bundle must be fully-rendered HTML (no raw Jinja tokens)
    assert "{{" not in html
    assert "{%" not in html

    # Stable signals that rendering logic exists
    assert "d3.forceSimulation" in html
    assert "<svg" in html
