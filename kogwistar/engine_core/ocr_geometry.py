"""Typed geometry records used by OCR document models."""

from pydantic import BaseModel, Field
from pydantic_extension.model_slicing import DtoType, ModeSlicingMixin


class box_2d(BaseModel):
    box_2d: list[int] = Field(description="box y min, x min, y max and x max")
    label: str = Field(description="text in the box")
    id: int = Field(description="id of the text box in the page, autoincrement from 0")


class NonText_box_2d(ModeSlicingMixin, BaseModel):
    """Recognised meaningful objects other than OCR characters, including figures."""

    description: DtoType[str] = Field(
        description="the description or summary of the non-OCR object"
    )
    box_2d: DtoType[list[int]] = Field(description="box y min, x min, y max and x max")
    id: int = Field(
        description="per page unique number of the cluster, starting from 0"
    )


class TextCluster(ModeSlicingMixin, BaseModel):
    """A text cluster with spatial information."""

    text: DtoType[str] = Field(description="the text content of the text cluster")
    bb_x_min: DtoType[float] = Field(
        description="the bounding box x min in pixel coordinate of the text_cluster. "
    )
    bb_x_max: DtoType[float] = Field(
        description="the bounding box x max in pixel coordinate of the text_cluster. "
    )
    bb_y_min: DtoType[float] = Field(
        description="the bounding box y min in pixel coordinate of the text_cluster. "
    )
    bb_y_max: DtoType[float] = Field(
        description="the bounding box y max in pixel coordinate of the text_cluster. "
    )
    cluster_number: int = Field(
        description="per page unique number of the cluster, starting from 0"
    )


class NonTextCluster(ModeSlicingMixin, BaseModel):
    """A non-text object with spatial information."""

    description: DtoType[str] = Field(
        description="the description or summary of the non-OCR object"
    )
    bb_x_min: DtoType[float] = Field(
        description="the bounding box x min in pixel coordinate of the non-OCR object. "
    )
    bb_x_max: DtoType[float] = Field(
        description="the bounding box x max in pixel coordinate of the non-OCR object. "
    )
    bb_y_min: DtoType[float] = Field(
        description="the bounding box y min in pixel coordinate of the non-OCR object. "
    )
    bb_y_max: DtoType[float] = Field(
        description="the bounding box y max in pixel coordinate of the non-OCR object. "
    )
    cluster_number: int = Field(
        description="per page unique number of the cluster, starting from 0"
    )
