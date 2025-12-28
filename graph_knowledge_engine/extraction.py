from graph_knowledge_engine.models import Span, Document
from graph_knowledge_engine.typing_interfaces import EngineLike
from typing import Type

class BaseDocValidator:
    def validate_span(self, span: Span, doc_id: str | None = None, doc: Document | None = None, engine: EngineLike | None = None):
        if (doc is not None) and doc_id is not None:
            if doc.id == doc_id:
                pass # ok they agree
            else:
                raise ValueError("Either doc and doc_id specified and they disagree")
        if doc is not None:
            pass
        else:
            if doc_id is None:
                # unreachable
                pass
            else:
                if engine is None:
                    raise ValueError("Engine is requried to resolve doc_id")
                else:
                    doc = engine.get_document(doc_id)
        if not doc:
            raise RuntimeError("fail to resolve document")
        excerpt_from_span = doc.get_content_by_span(span)
        return excerpt_from_span == span.excerpt
            
class PlainTextDocSpanValidator(BaseDocValidator):
    def validate_span(self, span: Span, doc_id: str | None = None, doc: Document | None = None, engine: EngineLike | None = None):
        return super().validate_span(span=span, doc_id = doc_id, doc = doc, engine = engine)
        if (doc is not None) and doc_id is not None:
            raise ValueError("Either doc or doc_id can be non None")
        if doc is not None:
            pass
        else:
            if doc_id is None:
                # unreachable
                pass
            else:
                if engine is None:
                    raise ValueError("Engine is requried to resolve doc_id")
                else:
                    doc = engine.get_document(doc_id)
        if not doc:
            raise RuntimeError("fail to resolve document")
        
        pass
        
    
    pass

class OcrDocSpanValidator(BaseDocValidator):
    def validate_span(self, span: Span, doc_id: str | None = None, doc: Document | None = None, engine: EngineLike | None = None):
        return super().validate_span(span=span)
        
    pass