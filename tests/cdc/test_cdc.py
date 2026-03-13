def test_dump_cdc_as_tool_not_temp_path():
    import os
    import pathlib

    out_dir = str(pathlib.Path("./empty_cdc_streamer").absolute())

    os.system(
        " ".join(
            [
                "python graph_knowledge_engine/utils/kge_debug_dump.py",
                "bundle",
                "--template",
                "./graph_knowledge_engine/templates/d3.html",
                "--out-dir",
                out_dir,
                "--empty",
                "--cdc-ws-url",
                "ws://127.0.0.1:8787/changes/ws",
            ]
        )
    )
    os.startfile(out_dir)
