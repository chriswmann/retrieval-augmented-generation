import lancedb

from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from lancedb.embeddings.sentence_transformers import SentenceTransformerEmbeddings
from wikipediaapi import Wikipedia


MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
# MODEL_NAME: str = "BAAI/bge-m3"

model_registry = get_registry().get("sentence-transformers")
model: SentenceTransformerEmbeddings = model_registry.create(
    name=MODEL_NAME, device="cpu"
)


class Document(LanceModel, arbitrary_types_allowed=True):
    text: str = model.SourceField()
    vector: Vector(384) = model.VectorField()  # type: ignore[report-invalid-type-form]
    category: str


def main() -> None:
    wiki: Wikipedia = Wikipedia("RAGBot9000", "en")
    docs = [
        {"text": x, "category": "cat"}
        for x in [wiki.page("Maru (cat)").text.split("\n\n")]
    ]
    docs += [
        {"text": x, "category": "painting"}
        for x in [wiki.page("Venus_Anadyomene_(Titian)").text.split("\n\n")]
    ]
    uri = "data/lancedb"
    db = lancedb.connect(uri)
    tbl_docs = db.create_table("documents", schema=Document, exist_ok=True)

    tbl_docs.add(docs)


if __name__ == "__main__":
    main()
