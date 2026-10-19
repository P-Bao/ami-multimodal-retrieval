from pydantic import BaseModel, Field


class ChatMessages(BaseModel):
    role: str
    content: str


class RAGRequest(BaseModel):
    messages: list[ChatMessages] | None = Field(default=None)
    top_k: int | None = Field(default=None, ge=1, le=200)
    include_references: bool = True


class RetrievedDocV2(BaseModel):
    text: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)
    reference_id: str | None = None
    doc: dict | None = None
    modality: str | None = None
    artifact_url: str | None = None
    table_body: str | None = None
    page: int | None = None
    caption: str | None = None


class RetrievalResponseV2(BaseModel):
    query: str
    documents: list[RetrievedDocV2] = Field(default_factory=list)
    references: list[dict] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)
