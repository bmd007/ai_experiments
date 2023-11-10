import asyncio

from langchain.chat_models import ChatOpenAI

api_base = "http://0.0.0.0:8001/v1"


async def main():
    chat = ChatOpenAI(temperature=0.0, openai_api_base=api_base, model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
    message = await chat.apredict(text="what is going on")
    print(message)


asyncio.run(main())
