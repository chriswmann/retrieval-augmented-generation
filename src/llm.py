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
    """Stream a chat from Llama with a nice 'real-time' output to the console.

    We use the AsyncClient to get a stream of responses from the Llama model, partly
    because the resulting effect is similar to real services such as chat.openai.com
    but primarily because this helps mask the latency of the model, since it is
    being run locally on a CPU.

    Parameters
    ----------
    async_client : AsyncClient
        The asynchronous client used to communicate with the Llama model.
    chat_content : str
        The content of the chat message to send to the model.

    Returns
    -------
    None
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
