import asyncio
import warnings

import lancedb

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from ollama import AsyncClient
from wikipediaapi import Wikipedia

from rag_mvp.llm import chat, prompt_format, system_prompt

MODEL_NAME: str = "BAAI/bge-m3"

warnings.simplefilter(action="ignore", category=FutureWarning)
model = (
    get_registry().get("sentence-transformers").create(name=MODEL_NAME, device="cpu")
)


class Document(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()  # type: ignore[report-invalid-type-form]
    category: str


def generate_documents(
    wiki: Wikipedia, page_title: str, page_cateogry: str
) -> list[dict[str, str]]:
    return [
        {"text": x[0], "category": page_cateogry}
        for x in [wiki.page(page_title).text.split("\n\n")]
    ]


def get_db(uri: str) -> lancedb.DBConnection:
    return lancedb.connect(uri)


def get_table(
    db: lancedb.DBConnection,
    table_name: str,
    schema: type[LanceModel],
    drop_table: bool = False,
    exist_ok: bool = True,
) -> lancedb.table.Table:
    if drop_table:
        db.drop_table(table_name)
    return db.create_table(table_name, schema=schema, exist_ok=exist_ok)


def build_prompt(sys_prompt: str, query: str, rag_text: str) -> str:
    formatted_system_prompt: str = sys_prompt.format(rag_text)

    return prompt_format.format(formatted_system_prompt, query)


def main() -> None:
    wiki: Wikipedia = Wikipedia("RAGBot9000", "en")
    maru_page_title: str = "Maru (cat)"
    maru_page_category: str = "cat"
    docs: list[dict[str, str]] = generate_documents(
        wiki, maru_page_title, maru_page_category
    )
    art_page_title: str = "Venus Anadyomene (Titian)"
    art_page_category: str = "painting"
    docs += generate_documents(wiki, art_page_title, art_page_category)

    election_page_title: str = "2024_Indian_general_election"
    election_page_category: str = "politics"
    docs += generate_documents(wiki, election_page_title, election_page_category)

    db_uri = "./data/lancedb"
    db: lancedb.DBConnection = get_db(uri=db_uri)
    table: lancedb.table.Table = get_table(
        db=db, table_name="docs", schema=Document, drop_table=True, exist_ok=True
    )
    table.add(docs)

    query: str = "Who won the election in India in 2024?"

    rag_result: LanceModel = table.search(query).limit(1).to_pydantic(Document)[0]
    rag_text: str = rag_result.text  # type: ignore[report-attribute-access-issue]

    async_client: AsyncClient = AsyncClient()

    prompt: str = build_prompt(sys_prompt=system_prompt, query=query, rag_text=rag_text)

    asyncio.run(chat(async_client=async_client, chat_content=prompt))


if __name__ == "__main__":
    main()
