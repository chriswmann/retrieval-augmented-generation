import warnings

import lancedb

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from wikipediaapi import Wikipedia

MODEL_NAME: str = "BAAI/bge-m3"
# MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

warnings.simplefilter(action="ignore", category=FutureWarning)
model = (
    get_registry().get("sentence-transformers").create(name=MODEL_NAME, device="cpu")
)


class Document(LanceModel, arbitrary_types_allowed=True):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()  # type: ignore[report-invalid-type-form]
    category: str


def main() -> None:
    wiki: Wikipedia = Wikipedia("RAGBot9000", "en")
    docs = [
        {"text": x[0], "category": "cat"}
        for x in [wiki.page("Maru (cat)").text.split("\n\n")]
    ]
    docs += [
        {"text": x[0], "category": "painting"}
        for x in [wiki.page("Venus_Anadyomene_(Titian)").text.split("\n\n")]
    ]
    uri = "./data/lancedb"
    db: lancedb.DBConnection = lancedb.connect(uri)
    table = db.create_table("documents", schema=Document, exist_ok=True)
    table.add(docs)

    query = "Japenese cat"

    actual = table.search(query).limit(1).to_pydantic(Document)[0]
    print(actual.text)  # type: ignore[report-attribute-access-issue]


if __name__ == "__main__":
    main()
