import asyncio
import warnings

import lancedb

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from ollama import AsyncClient
from wikipediaapi import Wikipedia

from rag_mvp.llm import chat

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

    db_uri = "./data/lancedb"
    db: lancedb.DBConnection = get_db(uri=db_uri)
    table: lancedb.table.Table = get_table(
        db=db, table_name="docs", schema=Document, drop_table=True, exist_ok=True
    )
    table.add(docs)

    query: str = "Who is Maru?"

    actual: LanceModel = table.search(query).limit(1).to_pydantic(Document)[0]
    actual_text: str = actual.text  # type: ignore[report-attribute-access-issue]

    prompt_format: str = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    system_prompt: str = f"""You are an extremely knowledgable expert.
    You always provide the most accurate information in a succinct manner.
    If you don't know the answer to something, you are honest about it.

    Here is some additional information that might be helpful to you:

        {actual_text}

    Do not quote this text verbatim and do not explicitly mention the source.

    If you are asked a question, start your response by repeating the question (in a
    concise form) and then provide your answer.
    """

    prompt: str = prompt_format.format(system_prompt, query)

    async_client: AsyncClient = AsyncClient()

    asyncio.run(chat(async_client=async_client, chat_content=prompt))


if __name__ == "__main__":
    main()
