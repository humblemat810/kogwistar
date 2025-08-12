import logging
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Document
from graph_knowledge_engine.visualization.basic_visualization import Visualizer
from joblib import Memory
import pathlib
def test_pretty_print_graph(engine: GraphKnowledgeEngine):
    import os
    
    doc_content = (
        "Photosynthesis is a process used by plants to convert light energy into chemical energy. "
        "Chlorophyll is the molecule that absorbs sunlight. "
        "Plants perform photosynthesis in their leaves."
    )
    doc = Document(
        content=doc_content,
        type="ocr",
        metadata={"source": "test"},
        processed=False
    )

    # cache ONLY the pure extraction on the doc content

    location=os.path.join(".cache", "test", pathlib.Path(__file__).parts[-1], "_extract_only")
    os.makedirs(location, exist_ok = True)
    memory = Memory(location=location, verbose=0)
    @memory.cache
    def _extract_only(content: str):
        return engine.extract_graph_with_llm(content=content)

    extracted = _extract_only(doc.content)
    parsed = extracted["parsed"]
    visualiser = Visualizer(engine=engine)
    # persist deterministically (choose replace/append/skip-if-exists)
    out = engine.persist_graph_extraction(document=doc, parsed=parsed, mode="replace")

    txt = visualiser.pretty_print_graph(by_doc_id=doc.id, include_refs=True)
    print(txt)
    logging.info(txt)
    