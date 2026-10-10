from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DocumentType = Literal["invoice", "contract"]


class DocumentSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1)
    license: str = Field(min_length=1)
    source_document_id: str | None = None


class PageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    image_path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class EntityAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    bbox: tuple[int, int, int, int]

    @model_validator(mode="after")
    def validate_bbox(self):
        x1, y1, x2, y2 = self.bbox

        if x1 < 0 or y1 < 0:
            raise ValueError("Bounding box coordinates cannot be negative")

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box must have positive width and height")

        return self


class RiskAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_type: str = Field(min_length=1)
    severity: Literal["low", "medium", "high", "critical"]
    evidence: str = Field(min_length=1)
    label_source: Literal["human", "synthetic_ground_truth", "rule"]


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    document_type: DocumentType
    source: DocumentSource
    split_group_id: str = Field(min_length=1)
    pages: list[PageMetadata] = Field(min_length=1)
    entities: list[EntityAnnotation] = Field(default_factory=list)
    risks: list[RiskAnnotation] = Field(default_factory=list)
    annotation_version: str = "0.1"

    @model_validator(mode="after")
    def validate_page_references(self):
        page_numbers = [page.page_number for page in self.pages]

        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("Page numbers must be unique within a document")

        available_pages = set(page_numbers)

        for entity in self.entities:
            if entity.page_number not in available_pages:
                raise ValueError(
                    f"Entity references missing page {entity.page_number}"
                )

            page = next(
                page for page in self.pages
                if page.page_number == entity.page_number
            )
            x1, y1, x2, y2 = entity.bbox

            if x2 > page.width or y2 > page.height:
                raise ValueError(
                    "Entity bounding box exceeds the page dimensions"
                )

        return self
