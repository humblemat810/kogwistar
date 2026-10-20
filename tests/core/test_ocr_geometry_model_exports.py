from kogwistar.engine_core.models import (
    NonText_box_2d,
    NonTextCluster,
    OCRClusterResponse,
    OCRClusterResponseBc,
    SplitPage,
    SplitPageMeta,
    TextCluster,
    box_2d,
)
from kogwistar.engine_core.ocr_geometry import (
    NonText_box_2d as OcrNonTextBox2d,
    NonTextCluster as OcrNonTextCluster,
    TextCluster as OcrTextCluster,
    box_2d as OcrBox2d,
)
from kogwistar.engine_core.ocr_page_models import (
    OCRClusterResponse as OcrClusterResponse,
    OCRClusterResponseBc as OcrClusterResponseBc,
    SplitPage as OcrSplitPage,
    SplitPageMeta as OcrSplitPageMeta,
)


def test_ocr_geometry_models_keep_legacy_import_identity() -> None:
    assert box_2d is OcrBox2d
    assert NonText_box_2d is OcrNonTextBox2d
    assert TextCluster is OcrTextCluster
    assert NonTextCluster is OcrNonTextCluster


def test_ocr_page_models_keep_legacy_import_identity() -> None:
    assert OCRClusterResponse is OcrClusterResponse
    assert OCRClusterResponseBc is OcrClusterResponseBc
    assert SplitPage is OcrSplitPage
    assert SplitPageMeta is OcrSplitPageMeta


def test_ocr_geometry_models_validate_typed_fields() -> None:
    box = box_2d(box_2d=[1, 2, 30, 40], label="heading", id=0)
    text = TextCluster(
        text="heading",
        bb_x_min=2.0,
        bb_x_max=40.0,
        bb_y_min=1.0,
        bb_y_max=30.0,
        cluster_number=0,
    )

    assert box.box_2d == [1, 2, 30, 40]
    assert text.text == "heading"
    assert text.cluster_number == 0


def test_split_page_serializer_keeps_requested_reading_order() -> None:
    clusters = [
        TextCluster(
            text=text,
            bb_x_min=0.0,
            bb_x_max=10.0,
            bb_y_min=float(index),
            bb_y_max=float(index + 1),
            cluster_number=index,
        )
        for index, text in enumerate(("first", "second"))
    ]
    page = SplitPage.model_construct(
        OCR_text_clusters=clusters,
        non_text_objects=[],
        meaningful_ordering=[1, 0],
        contains_table=False,
        printed_page_number=None,
        pdf_page_num=1,
        refined_version=None,
    )

    assert page.to_doc()["text"] == "second\nfirst"
