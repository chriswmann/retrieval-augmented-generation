from ollama import AsyncClient

prompt_format: str = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

system_prompt: str = """You are an extremely knowledgable expert.
You always provide the most accurate information in a succinct manner.
If you don't know the answer to something, you are honest about it.

Here is some additional information that might be helpful to you:

    {}

Do not quote this text verbatim and do not explicitly mention the source.

If you are asked a question, start your response by repeating the question (in a
concise form) and then provide your answer.
"""


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
