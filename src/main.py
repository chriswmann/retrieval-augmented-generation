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
    """Represents a document with text, vector, and category fields.

    LanceModel is a subclass of Pydantic's BaseModel, so we get data validation
    using this approach.
    """

    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()  # type: ignore[report-invalid-type-form]
    category: str


def generate_documents(
    wiki: Wikipedia, page_title: str, page_cateogry: str
) -> list[dict[str, str]]:
    """Generate documents from a Wikipedia page, using the wikipediaapi library.

    The page is identified via the page title at the moment. This is OK for the MVP
    but it would be better to use a more robust identifier, if possible (perhaps the URL).

    Parameters
    ----------
    wiki : Wikipedia
        wikipediaapi.Wikipedia object. TODO: consider using a protocol class to
        reduce the cohesion between this function and the wikipediaapi library.
    page_title : str
        The Wikipedia page title.
    page_cateogry : str
        A custom (not Wikipedia defined) category for the page.

    Returns
    -------
    list[dict[str, str]]
        Documents (paragraphs) with categories for the given Wikipedia page.
    """
    return [
        {"text": x[0], "category": page_cateogry}
        for x in [wiki.page(page_title).text.split("\n\n")]
    ]


def get_table(
    db: lancedb.DBConnection,
    table_name: str,
    schema: type[LanceModel],
    drop_table: bool = False,
    exist_ok: bool = True,
) -> lancedb.table.Table:
    """Retrieve or create a table in the LanceDB database.

    Note: this isn't production ready, as we wouldn't want to just drop
    an existing table, it just makes the MVP easier to setup and run.

    Parameters
    ----------
    db : lancedb.DBConnection
        The database connection object.
    table_name : str
        The name of the table to retrieve or create.
    schema : type[LanceModel]
        The schema defining the table structure.
    drop_table : bool, optional
        Whether to drop the table if it exists. Defaults to False.
    exist_ok : bool, optional
        Whether to ignore if the table exists. Defaults to True.

    Returns
    -------
    lancedb.table.Table
        The table object from the LanceDB database.
    """
    if drop_table:
        db.drop_table(table_name)
    return db.create_table(table_name, schema=schema, exist_ok=exist_ok)


def build_prompt(sys_prompt: str, query: str, rag_text: str) -> str:
    """Build a prompt for the language model using the provided query and context.

    This function formats the query and context into a single string that can be
    used as input for a language model.

    For the purposes of this MVP, the format is BAAI/bge-m3 specific.

    Parameters
    ----------
    query : str
        The user's query or question.
    context : str
        Additional context or background information to help the model generate a more accurate response.

    Returns
    -------
    str
        The formatted prompt string.
    """
    formatted_system_prompt: str = sys_prompt.format(rag_text)

    return prompt_format.format(formatted_system_prompt, query)


def main() -> None:
    """MVP entrypoint.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Wikipedia class requires a bot name but the choice is arbitrary for this MVP.
    wiki: Wikipedia = Wikipedia("RAGBot9000", "en")

    # Put some documents in the database that aren't related to the query, to test
    # that we don't retreive unrelated information.
    # TODO: add more documents to the database that are more closely related to the
    # query to test how relevant the retrieved information is.
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

    # Using local storage for simplicty in the MVP.
    db_uri = "./data/lancedb"
    db: lancedb.DBConnection = lancedb.connect(uri=db_uri)
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
