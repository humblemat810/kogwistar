"""Validated OCR response and persisted page models."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_extension.model_slicing import DtoType, ModeSlicingMixin
from pydantic_extension.model_slicing.mixin import DtoField

from .ocr_geometry import NonTextCluster, TextCluster


class OCRClusterResponse(ModeSlicingMixin, BaseModel):
    """Validated OCR output for one page, including reading order and quality."""

    OCR_text_clusters: DtoType[list[TextCluster]] = Field(
        description="the OCR text results. Share cluster number uniqueness with non-OCR objects. "
    )
    non_text_objects: DtoType[list[NonTextCluster]] = Field(
        description="the non-OCR object results. Share cluster number uniqueness with OCR texts. "
    )
    is_empty_page: DtoType[bool | None] = Field(
        default=False,
        description="true if the whole page is empty without recognisable text.",
    )
    printed_page_number: DtoType[str | None] = Field(
        description='the page number identified from OCR texts, can be in form of roman numerals such as "i", "ii", "iii", "iv"...; '
        'Arabic numeral such as 1, 2, 3... or letter such as "a", "b", "c"...\n'
        'Sometimes the are surrounded by symbols such as "- 1 -", "- 2 -"'
        r"Can be null/none if there is no page order assigned and printed and found in the scanned texts. Do not assign page number. Only use page number found."
    )
    meaningful_ordering: DtoType[list[int]] = Field(
        description="The correct meaningful ordering of the identified text clusters. Must cover all OCR_text_clusters once and only once. "
    )
    page_x_min: DtoType[float] = Field(
        description="the page x min in pixel coordinate. "
    )
    page_x_max: DtoType[float] = Field(
        description="the page x max in pixel coordinate. "
    )
    page_y_min: DtoType[float] = Field(
        description="the page y min in pixel coordinate. "
    )
    page_y_max: DtoType[float] = Field(
        description="the page y max in pixel coordinate. "
    )
    estimated_rotation_degrees: DtoType[float] = Field(
        description="the page estimated rotation degree using right hand rule. "
    )
    incomplete_words_on_edge: DtoType[bool] = Field(
        description="If there is any text being incomplete due to the scan does not scan the edges properly. "
    )
    incomplete_text: DtoType[bool] = Field(description="Any incomplete text")
    data_loss_likelihood: DtoType[float] = Field(
        description="The likelihood (range from 0.0 to 1.0 inclusive) that the page has lost information by missing the scan data on the edges of the page."
    )
    scan_quality: DtoType[Literal["low", "medium", "high"]] = Field(
        description="The image quality of the scan. All qualities exclude signatures. "
        '"low", "medium" or "high". '
        "low: text barely legible. medium: Legible with non smooth due to pixelation. high: texts are easily and highly identifiable. "
    )
    contains_table: DtoType[bool] = Field(
        description="Whether this page contains table. "
    )

    @model_validator(mode="after")
    def check_cluster_meaningful_ordering_agreement(self) -> Self:
        assert bool(self.is_empty_page) ^ (len(self.OCR_text_clusters) > 0), (
            f"is_empty_page value {self.is_empty_page} disagree with OCR_text_clusters len={len(self.OCR_text_clusters)}"
        )
        cluster_numbers = [
            item.cluster_number
            for item in self.non_text_objects + self.OCR_text_clusters
        ]
        if len(cluster_numbers) != len(set(cluster_numbers)):
            raise ValueError(
                "cluster number from non_text_objects block and ocr text blocks must be ALL distinct. "
            )
        if len(self.meaningful_ordering) != len(set(self.meaningful_ordering)):
            raise ValueError("meaningful_order must cover each text cluster at most once")
        return self


class OCRClusterResponseBc(OCRClusterResponse):
    """Backward-compatible response model retained for existing callers."""


class SplitPageMeta(BaseModel):
    ocr_model_name: str = Field(description="model that perform OCR")
    ocr_datetime: float = Field(description="unix timestamp when ocr is performed")
    ocr_json_version: str = Field(description="the model does the OCR")

    @field_validator("ocr_json_version", mode="before")
    def version_to_str(cls, value: object) -> str:
        return str(value)


class SplitPage(OCRClusterResponseBc):
    """Persisted OCR page with a stable legacy projection serializer."""

    pdf_page_num: int
    metadata: SplitPageMeta
    refined_version: OCRClusterResponse[DtoField] | None = Field(
        default=None,
        description="refined processed/ grouped/ merged version of ocr text clusters. ",
    )

    def model_dump(self, *arg: object, **kwarg: object) -> dict[str, object]:
        return self.to_doc()

    def dump_raw(self) -> dict[str, object]:
        return super(SplitPage, self).model_dump(exclude=["refined_version"])

    def dump_supercede_parse(self) -> dict[str, object]:
        return super(SplitPage, self).model_dump(
            exclude=["refined_version", "metadata"]
        )

    @model_validator(mode="after")
    def roundtrip_invariant(self, info: ValidationInfo) -> SplitPage:
        ctx = info.context or {}
        if ctx.get("_roundtrip_active", False):
            return self
        self.to_doc()
        nested_ctx = dict(ctx)
        nested_ctx["_roundtrip_active"] = True
        dumped = self.model_dump(mode="json")
        again = self.__class__.model_validate(dumped, context=nested_ctx)
        if again != self:
            raise ValueError(
                "Roundtrip invariant failed: dump->validate changed the model"
            )
        return self

    def to_doc(self) -> dict[str, object]:
        """Serialize a page in the legacy LLM payload shape and reading order."""
        target = self.refined_version or self
        ocr_cluster = target.OCR_text_clusters
        non_ocr_cluster = target.non_text_objects
        if target.contains_table:
            serialized_clusters: list[dict[str, object]] = []
            cluster_numbers = (item.cluster_number for item in ocr_cluster)
            assert len(set(cluster_numbers)) == len(ocr_cluster)
            cluster_lookup_by_number: dict[int, TextCluster | NonTextCluster] = {
                item.cluster_number: item
                for item in ocr_cluster + non_ocr_cluster
            }
            for cluster_number in target.meaningful_ordering:
                cluster = cluster_lookup_by_number.get(cluster_number)
                if cluster is None:
                    raise KeyError(f"{cluster_number} does not exist")
                cluster_dump: dict[str, object] = cluster.model_dump()
                cluster_dump.pop("cluster_number")
                serialized_clusters.append(cluster_dump)
            for cluster_number in set(cluster_numbers) - set(
                target.meaningful_ordering
            ):
                cluster_dump = cluster_lookup_by_number[cluster_number].model_dump()
                cluster_dump.pop("cluster_number")
                serialized_clusters.append(cluster_dump)
            return {
                "pdf_page_num": self.pdf_page_num,
                "printed_page_number": self.printed_page_number,
                "OCR_text_clusters": serialized_clusters,
                "contains_table": self.contains_table,
            }

        ocr_cluster_numbers = sorted(item.cluster_number for item in ocr_cluster)
        non_text_cluster_numbers = sorted(
            item.cluster_number for item in non_ocr_cluster
        )
        if ocr_cluster_numbers:
            start_number = min(ocr_cluster_numbers)
            assert start_number in [0, 1], (
                "only allow 0-indexed based or 1-indexed based cluster numbers"
            )
            expected_number = start_number
            index = 0
            while expected_number < start_number + len(
                ocr_cluster_numbers + non_text_cluster_numbers
            ):
                if (
                    index < len(ocr_cluster)
                    and expected_number == ocr_cluster[index].cluster_number
                ):
                    index += 1
                else:
                    raise Exception(
                        "expected index {expected_number} not found in both ocr_cluster nor signature_blocks"
                    )
                expected_number += 1

        sorted_text_clusters = sorted(
            ocr_cluster, key=lambda item: item.cluster_number
        )
        assert len(sorted_text_clusters) == len(
            {item.cluster_number for item in sorted_text_clusters}
        )
        is_normal = all(
            item.cluster_number == index
            for index, item in enumerate(sorted_text_clusters)
        )
        shift = 0
        if not is_normal:
            is_normal = all(
                item.cluster_number == index + 1
                for index, item in enumerate(sorted_text_clusters)
            )
            shift = -1
        assert set(target.meaningful_ordering) <= {
            item.cluster_number for item in self.OCR_text_clusters
        }
        if is_normal:
            texts = "\n".join(
                sorted_text_clusters[number + shift].text
                for number in target.meaningful_ordering
            )
        else:
            text_by_cluster_number = {
                item.cluster_number: item.text for item in sorted_text_clusters
            }
            texts = "\n".join(
                text_by_cluster_number[number]
                for number in target.meaningful_ordering
            )
        return {
            "pdf_page_num": self.pdf_page_num,
            "printed_page_number": self.printed_page_number,
            "text": texts,
        }
