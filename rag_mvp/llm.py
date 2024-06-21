from ollama import AsyncClient


async def chat(async_client: AsyncClient, chat_content: str) -> None:
    """
    Stream a chat from Llama using the AsyncClient.
    """
    message = {
        "role": "user",
        "content": chat_content,
    }
    async for part in await async_client.chat(
        model="llama3",
        messages=[message],  # type: ignore[report-argument-type]
        stream=True,  # type: ignore[report-general-type-issues]
    ):
        print(part["message"]["content"], end="", flush=True)

    print()
