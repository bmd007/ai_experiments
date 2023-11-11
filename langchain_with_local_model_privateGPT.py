from langchain import hub
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader
from langchain.llms import Ollama
from langchain.schema import LLMResult
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# load log file
loader = JSONLoader(file_path="/Users/mohami/workspacce/personal-repositories/ai_experiments/local_data/try.json",
                    jq_schema=".[].jsonPayload.message",
                    text_content=False)
data = loader.load()
# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=9000, chunk_overlap=500)
all_splits = text_splitter.split_documents(data)

# OllamaEmbeddings(base_url="http://0.0.0.0:8001", model="dolphin-2.2.1-mistral-7b.Q8_0")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)

# RAG prompt
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")


class GenerationStatisticsCallback(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(response.generations[0][0].generation_info)


callback_manager = CallbackManager(
    [
        StreamingStdOutCallbackHandler(),
        # GenerationStatisticsCallback()
    ]
)

llm = Ollama(
    base_url="http://localhost:8001",
    model="dolphin-2.2.1-mistral-7b.Q8_0",
    verbose=True,
    callback_manager=callback_manager,
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

question = "Any customer having problem with enrollment? show the corresponding customerId and other identifiers if any"
result = qa_chain({"query": question})
